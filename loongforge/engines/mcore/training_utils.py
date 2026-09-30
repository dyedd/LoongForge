# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Pretrain utilities."""

import os
import dataclasses
import gc
from datetime import datetime, timedelta
import logging
import sys
import re

try:
    from nvidia_resiliency_ext.inprocess import CallWrapper
except ImportError:
    CallWrapper = type(None)

from megatron.training.log_handler import CustomHandler

from typing import Optional
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

# Make default logging level INFO, but filter out all log messages not from MCore.
logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO)

import time

# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch
from collections import OrderedDict
from .checkpointing import load_checkpoint, _load_checkpoint_from_path
from megatron.core import mpu
from megatron.core.utils import (
    check_param_hashes_across_dp_replicas,
    get_model_config,
    StragglerDetector,
)
from megatron.core.num_microbatches_calculator import (
    get_num_microbatches,
    update_num_microbatches,
    get_current_global_batch_size,
    get_current_running_global_batch_size,
)
from megatron.core.fp8_utils import correct_amax_history_if_needed
from megatron.core.transformer.module import Float16Module
from megatron.core.enums import ModelType
from megatron.core import mpu, tensor_parallel
from megatron.training.utils import to_empty_if_meta_device
from megatron.core.distributed import (
    DistributedDataParallelConfig,
    TorchFullyShardedDataParallelConfig,
)
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

from megatron.core.distributed import (
    DistributedDataParallel as DDP,
    finalize_model_grads,
)
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel as megatron_FSDP,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.core.rerun_state_machine import get_rerun_state_machine, RerunDataIterator, RerunState
from megatron.core.transformer.moe import upcycling_utils
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.training.global_vars import get_energy_monitor

from megatron.core.parallel_state import update_pg_timeout

from megatron.training import (
    get_signal_handler,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    print_rank_0,
    print_rank_last,
    ft_integration,
)
from megatron.training.initialize import (
    write_args_to_tensorboard,
    set_jit_fusion_options,
)
from .checkpointing import (
    load_checkpoint,
    save_checkpoint,
    checkpoint_exists,
)
from megatron.training.utils import (
    calc_params_l2_norm,
    report_memory,
    unwrap_model,
    update_use_dist_ckpt,
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
    is_last_rank,
)
from megatron.training.theoretical_memory_usage import report_theoretical_memory
from megatron.training.async_utils import maybe_finalize_async_save
from megatron.training.training import (
    append_to_progress_log,
    print_datetime,
    build_train_valid_test_data_iterators,
    evaluate_and_print_results,
    num_floating_point_operations,
    get_start_time_from_progress_log,
    get_optimizer_param_scheduler,
    preprocess_common_state_dict,
    should_disable_forward_pre_hook,
    disable_forward_pre_hook,
    enable_forward_pre_hook,
    dummy_train_step,
    post_training_step_callbacks,
    checkpoint_and_decide_exit,
)
from loongforge.models.common.peft.lora import LoRA, VLMLoRA
from loongforge.models.common.peft.utils import apply_peft_transformation
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from dataclasses import asdict
from loongforge.engines.mcore import get_args, constants, global_vars
from .initialize import initialize_loongforge_megatron
from loongforge.engines.mcore.parallel.dp_balance.train_hooks import (
    train_step_decorator,
    train_log_decorator
)


# Add project root to Python path
import sys
import os
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add tools directory to Python path
tools_path = os.path.join(project_root, "tools")
if tools_path not in sys.path:
    sys.path.insert(0, tools_path)

from dist_checkpoint.checkpoint.hf_checkpoint_loader import load_hf_checkpoint_online
from dist_checkpoint.checkpoint.hf_checkpoint_saver import save_hf_checkpoint_online

try:
    from inspector.hooks import register_hooks

    HAS_INSPECTOR = True
except ImportError:
    HAS_INSPECTOR = False

stimer = StragglerDetector()

# ---------------------------------------------------------------------------
# CUDA graph registries moved to optimizations/cuda_graph_registry.py
# ---------------------------------------------------------------------------
from loongforge.engines.mcore.optimizations.cuda_graph_registry import (
    _FULL_ITER_GRAPH_REGISTRY,
    register_full_iteration_graph_wrapper,
    get_full_iteration_graph_wrapper,
    _PER_MICROBATCH_GRAPH_REGISTRY,
    register_per_microbatch_graph_wrapper,
    get_per_microbatch_graph_wrapper,
)


from loongforge.engines.mcore.model_setup import (
    is_hf_checkpoint,
    _resolve_convert_pp_layout,
    update_ema,
    freeze_parameters,
    check_vlm_peft_config,
    add_hooks,
    _p2p_embedding_weights_for_mtp,
    get_model,
    setup_model_and_optimizer,
)


from loongforge.training.mcore_runner import enable_memory_history_record



def add_hooks(model, args, prefix):
    """add hooks for model, only enable hooks when open the switch: enable-log-tensor"""
    from loongforge.engines.mcore.model_setup import add_hooks as _add_hooks
    return _add_hooks(model, args, prefix)


def pretrain(
    train_args,
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    process_non_loss_data_func=None,
    extra_args_provider=None,
    args_defaults={},
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    non_loss_data_func=None,
    store=None,
    inprocess_call_wrapper: Optional[CallWrapper] = None,
):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the model using the forward_step_func.

    Args:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
        get_embedding_ranks (TODO):
        get_position_embedding_ranks (TODO):
        non_loss_data_func (callable): A custom function to call during evaluation.
            It can run e.g. benchmarks.
        store: an optional instance of torch.distributed.Store, to be used by
            torch.distributed.init_process_group
        inprocess_call_wrapper: an optional instance of inprocess.CallWrapper,
            it is automatically injected when in-process restart is in use
    """

    if inprocess_call_wrapper is not None:
        iteration = inprocess_call_wrapper.iteration
        store = torch.distributed.PrefixStore(str(iteration), store)

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_loongforge_megatron(
        args=train_args,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        store=store,
    )

    args = get_args()
    timers = get_timers()

    if args.log_progress:
        append_to_progress_log("Starting job")

    # Initialize fault tolerance
    # NOTE: ft_integration functions other than `setup` are no-op if the FT is not initialized
    if args.enable_ft_package:
        ft_integration.setup(args)
        ft_integration.maybe_setup_simulated_fault()

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor(
        [_TRAIN_START_TIME], dtype=torch.double, device="cuda"
    )
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)

    _TRAIN_START_TIME = start_time_tensor.item()

    print_rank_0(
        "time to initialize megatron (seconds): {:.3f}".format(
            time.time() - _TRAIN_START_TIME
        )
    )
    print_datetime("after megatron is initialized")

    # enable memory histroy record
    if hasattr(args, "record_memory_history") and args.record_memory_history:
        enable_memory_history_record(args.memory_snapshot_path)

    # Context used for persisting some state between checkpoint saves.
    if args.non_persistent_ckpt_type == "local":
        try:
            from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import (
                LocalCheckpointManager,
            )
            from nvidia_resiliency_ext.checkpointing.local.replication.group_utils import (
                parse_group_sequence,
                GroupWrapper,
            )
            from nvidia_resiliency_ext.checkpointing.local.replication.strategies import (
                CliqueReplicationStrategy,
            )
        except ModuleNotFoundError:
            raise RuntimeError(
                "The 'nvidia_resiliency_ext' module is required for local "
                "checkpointing but was not found. Please ensure it is installed."
            )

        if args.replication:
            repl_strategy = CliqueReplicationStrategy.from_replication_params(
                args.replication_jump, args.replication_factor
            )
        else:
            repl_strategy = None

        checkpointing_context = {
            "local_checkpoint_manager": LocalCheckpointManager(
                args.non_persistent_local_ckpt_dir, repl_strategy=repl_strategy
            )
        }
    else:
        checkpointing_context = {}

    # Model, optimizer, and learning rate.
    timers("model-and-optimizer-setup", log_level=0).start(barrier=True)
    model, ema, optimizer, opt_param_scheduler, peft_class = setup_model_and_optimizer(
        model_provider, model_type, checkpointing_context=checkpointing_context
    )

    if args.enable_log_tensor and HAS_INSPECTOR:
        # following only for trace tesnors
        # debug infos
        unwrap_models = unwrap_model(model)
        index = 0
        for tmp_model in unwrap_models:
            prefix = "chunk" + str(index)
            add_hooks(tmp_model, args, prefix)
            index += 1

    timers("model-and-optimizer-setup").stop()
    print_datetime("after model, optimizer, and learning rate scheduler are built")
    config = get_model_config(model[0])

    # Data stuff.
    timers("train/valid/test-data-iterators-setup", log_level=0).start(barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            iterators = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider,
                vp_stage=i,
            )
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator = (
            build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
        )

    timers("train/valid/test-data-iterators-setup").stop()
    print_datetime("after dataloaders are built")

    # Print setup timing.
    print_rank_0("done with setup ...")
    timers.log(
        ["model-and-optimizer-setup", "train/valid/test-data-iterators-setup"],
        barrier=True,
    )

    wandb_writer = get_wandb_writer()
    if wandb_writer:
        # Add job name to the wandb config to make it easier to run more singleton dependency jobs.
        wandb_writer.config.update(
            {"slurm_job_name": os.getenv("SLURM_JOB_NAME", "N/A")}
        )

    if not args.skip_train:
        print_rank_0("training ...")

        if args.dataloader_type == "cyclic" and args.retro_project_dir:
            assert args.retro_cyclic_train_iters is not None
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0
        if args.do_train and args.train_iters > 0:
            iteration, num_floating_point_operations_so_far = train(
                forward_step_func=forward_step_func,
                model=model,
                ema=ema,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
                train_data_iterator=train_data_iterator,
                valid_data_iterator=valid_data_iterator,
                process_non_loss_data_func=process_non_loss_data_func,
                config=config,
                checkpointing_context=checkpointing_context,
                non_loss_data_func=non_loss_data_func,
            )

        print_datetime("after training is done")

        if args.save and iteration != 0 and iteration % args.save_interval != 0:
            save_checkpoint(
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
                num_floating_point_operations_so_far=num_floating_point_operations_so_far,
                checkpointing_context=checkpointing_context,
                train_data_iterator=train_data_iterator,
                preprocess_common_state_dict_fn=preprocess_common_state_dict,
                peft_class=peft_class,
            )

            if args.enable_ema and ema is not None:
                save_checkpoint(
                    iteration=iteration,
                    model=ema,
                    optimizer=None,
                    opt_param_scheduler=None,
                    num_floating_point_operations_so_far=num_floating_point_operations_so_far,
                    save_arg="save_ema",
                    peft_class=peft_class,
                )

    else:
        print_rank_0("skipping training (--skip-train is on) ...")

        iteration = args.iteration

    if args.do_valid:
        prefix = f"iteration {iteration} on validation set"
        evaluate_and_print_results(
            prefix,
            forward_step_func,
            valid_data_iterator,
            model,
            iteration,
            process_non_loss_data_func,
            config,
            verbose=True,
            write_to_tensorboard=not args.skip_train,
            non_loss_data_func=non_loss_data_func,
        )

    if args.do_test:
        prefix = f"iteration {iteration} on test set"
        evaluate_and_print_results(
            prefix,
            forward_step_func,
            test_data_iterator,
            model,
            iteration,
            process_non_loss_data_func,
            config,
            verbose=True,
            write_to_tensorboard=not args.skip_train,
            non_loss_data_func=non_loss_data_func,
        )
    
    # Save HF checkpoint at the end of training if --save-hf is enabled
    save_hf_enabled = getattr(args, 'save_hf', 'false').lower() == 'true'
    if save_hf_enabled and iteration == args.train_iters:
        # Set default save_hf_path if not specified
        if getattr(args, 'save_hf_path', None) is None and args.save is not None:
            args.save_hf_path = os.path.join(args.save, "release_hf_weights/")
        if hasattr(config, 'pipeline_model_parallel_layout'):
            args.convert_pp_layout = config.pipeline_model_parallel_layout

        # Save HF checkpoint
        save_hf_checkpoint_online(model, args)
        torch.distributed.barrier()

    wandb_writer = get_wandb_writer()
    if wandb_writer:
        wandb_writer.finish()

    ft_integration.on_checkpointing_start()
    maybe_finalize_async_save(blocking=True, terminate=True)
    ft_integration.on_checkpointing_end(is_async_finalization=True)

    ft_integration.shutdown()

def check_vlm_peft_config(model_config):
    from loongforge.engines.mcore.model_setup import check_vlm_peft_config as _check
    return _check(model_config)


# print_module_param_dtypes: deleted, import from upstream Megatron
from megatron.training.training import print_module_param_dtypes

def compute_throughputs_and_append_to_progress_log(
    iteration, num_floating_point_operations_so_far
):
    """Compute throughputs and append to progress log."""
    args = get_args()
    if args.save is None:
        return

    # Compute job throughput.
    # args.num_floating_point_operations_so_far keeps track of floating-point operations
    # completed at the start of job.
    global _TRAIN_START_TIME
    job_throughput = (
        num_floating_point_operations_so_far - args.num_floating_point_operations_so_far
    ) / ((time.time() - _TRAIN_START_TIME) * 10**12 * args.world_size)

    # Compute cumulative throughput since jobs of this world size were launched.
    # `get_start_time_from_progress_log` returns start time and number of floating-point
    # operations of first job of this world size.
    start_time, start_num_floating_point_operations = get_start_time_from_progress_log()
    elapsed_time = (datetime.now() - start_time).total_seconds()
    cumulative_throughput = (
        num_floating_point_operations_so_far - start_num_floating_point_operations
    ) / (elapsed_time * 10**12 * args.world_size)

    tokens_so_far = args.consumed_train_samples * args.seq_length
    saved_ckpt_prefix = (
        "Saving async checkpoint" if args.async_save else "Saved checkpoint"
    )
    append_to_progress_log(
        f"{saved_ckpt_prefix}\tIteration: {iteration}\t"
        f"Job throughput: {job_throughput:.1f} TFLOP/s/GPU\t"
        f"Cumulative throughput: {cumulative_throughput:.1f} TFLOP/s/GPU\t"
        f"Floating-point operations: {num_floating_point_operations_so_far:.2e}\t"
        f"Tokens (in billions): {tokens_so_far / 10**9:.2f}"
    )


def save_checkpoint_and_time(
    iteration,
    model,
    ema,
    optimizer,
    opt_param_scheduler,
    num_floating_point_operations_so_far,
    checkpointing_context,
    non_persistent_ckpt=False,
    train_data_iterator=None,
):
    """Save checkpoint and time."""
    args = get_args()
    timers = get_timers()
    energy_monitor = get_energy_monitor()

    # Stop timer to get accurate train interval time and exclude checkpointing duration
    timers("interval-time").stop()
    energy_monitor.pause()
    # Extra barrier is added to make sure all ranks report the max time.
    timer_key = (
        "save-checkpoint-non-persistent" if non_persistent_ckpt else "save-checkpoint"
    )
    timers(timer_key, log_level=0).start(barrier=True)

    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model)

    save_checkpoint(
        iteration=iteration,
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        num_floating_point_operations_so_far=num_floating_point_operations_so_far,
        checkpointing_context=checkpointing_context,
        non_persistent_ckpt=non_persistent_ckpt,
        train_data_iterator=train_data_iterator,
        preprocess_common_state_dict_fn=preprocess_common_state_dict,
    )
    if args.fp8:
        # Run garbage collection after checkpoint saving to free memory from
        # dequantized bf16 tensors that were temporarily created during fp8
        # model checkpoint saving.
        gc.collect()
    if should_disable_forward_pre_hook(args):
        enable_forward_pre_hook(model)

    if args.enable_ema and ema is not None:
        save_checkpoint(
            iteration=iteration,
            model=ema,
            optimizer=None,
            opt_param_scheduler=None,
            num_floating_point_operations_so_far=num_floating_point_operations_so_far,
            save_arg="save_ema",
        )

    timers(timer_key).stop(barrier=True)
    timers.log([timer_key])

    if args.log_progress and not non_persistent_ckpt:
        compute_throughputs_and_append_to_progress_log(
            iteration, num_floating_point_operations_so_far
        )

    # Recover timing
    energy_monitor.resume()
    timers("interval-time", log_level=0).start(barrier=True)

from loongforge.engines.mcore.train_step import train_step, stimer
from loongforge.training.mcore_runner import (
    pretrain,
    compute_throughputs_and_append_to_progress_log,
    save_checkpoint_and_time,
    training_log,
    train,
    _TRAIN_START_TIME,
)

# dump_model_input_example_once and related module vars moved to train_step.py
from loongforge.engines.mcore.train_step import (
    dump_model_input_example_once,
    _PRINTED_MODEL_INPUT_EXAMPLE,
    _SAMPLE_DUMP_ANSI_COLOR,
    _SAMPLE_DUMP_ANSI_RESET,
)

