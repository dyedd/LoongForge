# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""SFT dataloader utilities: iterators, collators, and cyclic data builders."""

import logging
import os

import torch
from torch.utils.data import DataLoader
from transformers.utils import PaddingStrategy

from datasets.distributed import split_dataset_by_node

from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.legacy.data.data_samplers import MegatronPretrainingRandomSampler

from loongforge.engines.mcore import get_args, get_model_config, get_tokenizer, constants
from loongforge.data import DataCollatorForSupervisedDataset
from loongforge.engines.mcore.tokenizer import AutoTokenizerFromHF
from loongforge.engines.mcore.checkpointing import get_checkpoint_name, read_tracker_iteration

logger = logging.getLogger(__name__)


def get_dataset_blend_from_list(
    dataset_names: Optional[List[str]],
) -> Optional[List[str]]:
    """get dataset from list"""
    if dataset_names is None:
        return None

    return [_dataset_name.strip() for _dataset_name in dataset_names]


def _cyclic_iter(iter):
    """cyclic iteration"""
    while True:
        for x in iter:
            yield x


def build_sft_data_collator(
    cls: Type[DataCollatorForSupervisedDataset], **kwargs
) -> DataCollatorForSupervisedDataset:
    """build data collator for sft"""
    args = get_args()
    tokenizer = get_tokenizer()

    assert isinstance(
        tokenizer, AutoTokenizerFromHF
    ), f"Only support HFTokenizer for sft, but got {args.tokenizer_type}."

    pad_to_multiple_of = 1
    # When using sequence parallel, sequence will further be split by TP size
    # When using context parallel, sequence is split by CP size as well
    pad_to_multiple_of *= (
        args.tensor_model_parallel_size if args.sequence_parallel else 1
    )
    pad_to_multiple_of *= (
        (2 * args.context_parallel_size) if args.context_parallel_size > 1 else 1
    )

    # https://github.com/NVIDIA/TransformerEngine/blob/v2.4/transformer_engine/pytorch/utils.py#L425
    # https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/gemm/cublaslt_gemm.cu#L151
    pad_to_multiple_of *= 128 if args.fp8 else 1
    if args.enable_chunkpipe and getattr(args, "sft_chunkpipe_mode", False):
        pad_to_multiple_of = args.chunksize

    padding = (
        PaddingStrategy.LONGEST
        if args.variable_seq_lengths
        else PaddingStrategy.MAX_LENGTH
    )

    # When chunkpipe is enabled, all base chunks are already padded to chunksize.
    # If SFT chunkpipe + MTP is enabled, the collator temporarily strips bridge
    # tokens, pads only the base part to pad_to_multiple_of, then appends bridge
    # tokens back.
    max_length = args.chunksize if args.enable_chunkpipe else args.seq_length

    if args.enable_chunkpipe and getattr(args, "sft_chunkpipe_mode", False):
        kwargs["chunkpipe_base_length"] = args.chunksize
        kwargs["chunkpipe_mtp_num_layers"] = args.mtp_num_layers or 0

    data_collator = cls(
        tokenizer=tokenizer.hf_tokenizer(),
        label_pad_token_id=constants.IGNORE_INDEX,
        pad_to_multiple_of=pad_to_multiple_of,
        padding=padding,
        max_length=max_length,
        **kwargs,
    )
    return data_collator


class _IterableWithState:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.step = 0
        self._iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self._iterator)
            self.step += 1
            return batch
        except StopIteration:
            self._iterator = iter(self.dataloader)
            # self.step = 0
            batch = next(self._iterator)
            self.step += 1
            return batch

    def save_state(self):
        """dataloader save state"""
        return {"step": self.step}

    def load_state(self, state):
        """dataloader load state"""
        target = state.get("step", 0)
        if target <= self.step:
            return
        for _ in range(target - self.step):
            next(self._iterator)
        self.step = target


class SavableCyclicIterator:
    """
    Cyclic iterator that:
      - exposes `.iterable` with save_state/load_state (via _IterableWithState)
      - yields batches infinitely
    Compatible with Megatron's maybe_save_dataloader_state().
    """

    def __init__(self, dataloader):
        self.iterable = _IterableWithState(dataloader)
        self._iterator = self._cyclic_iter(self.iterable)

    def _cyclic_iter(self, iterable_with_state):
        while True:
            for batch in iterable_with_state:
                yield batch

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def save_state(self):
        """dataloader save state"""
        return self.iterable.save_state()

    def load_state(self, state):
        """dataloader load state"""
        return self.iterable.load_state(state)


class SavableCyclicIteratorWithPreprocessor:
    """
    Cyclic iterator that applies a preprocessor to each batch and supports
    save_state/load_state for checkpoint resumption.

    The preprocessor is applied after the _IterableWithState step counter
    increments, so each DataLoader batch = one step regardless of preprocessing.

    Compatible with Megatron's maybe_save_dataloader_state().
    """

    def __init__(self, dataloader, preprocessor=None):
        self.iterable = _IterableWithState(dataloader)
        self.preprocessor = preprocessor
        self._iterator = self._cyclic_iter(self.iterable)

    def _cyclic_iter(self, iterable_with_state):
        while True:
            for batch in iterable_with_state:
                if self.preprocessor is not None:
                    yield self.preprocessor(batch)
                else:
                    yield batch

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def save_state(self):
        """dataloader save state"""
        return self.iterable.save_state()

    def load_state(self, state):
        """dataloader load state"""
        return self.iterable.load_state(state)


def build_savable_dataloader_iter(dataloader, preprocessor=None):
    """Build a savable cyclic iterator with optional preprocessor.

    If args.dataloader_save is set, returns a SavableCyclicIteratorWithPreprocessor
    that supports save_state/load_state for checkpoint resumption.
    Otherwise returns a plain cyclic generator (no state tracking).

    Also restores dataloader state from a previous checkpoint if applicable.

    Args:
        dataloader: PyTorch DataLoader to wrap.
        preprocessor: Optional callable applied to each batch.

    Returns:
        A cyclic iterator (savable or plain).
    """
    from loongforge.engines.mcore import get_args, print_rank_0

    args = get_args()

    # Use args.dataloader_save if set; fall back to args.save so VLA trainers
    # that set dataloader_save = args.save in _ensure_megatron_defaults still
    # work even when Megatron re-parses args after initialize_megatron().
    dl_save = getattr(args, "dataloader_save", None) or getattr(args, "save", None)
    dl_load = getattr(args, "load", None)

    if dl_save is not None:
        train_iter = SavableCyclicIteratorWithPreprocessor(dataloader, preprocessor=preprocessor)

        # Restore dataloader state when resuming from a checkpoint.
        # When --no-load-optim/--finetune resets args.iteration to 0, we cannot
        # rely on it to find the correct checkpoint directory. Instead, scan for
        # the latest checkpoint iteration that contains a dataloader state file.
        if dl_load is not None:

            dp_rank = mpu.get_data_parallel_rank()
            restored = False

            # Determine the checkpoint iteration using the same logic as
            # Megatron's load_checkpoint: read latest_checkpointed_iteration.txt.
            # This ensures the dataloader state matches the actually-loaded
            # model checkpoint, rather than blindly picking the largest iter.
            candidates = []
            iteration = getattr(args, "iteration", 0) or 0
            if iteration > 0:
                candidates.append(iteration)

            tracker_result = read_tracker_iteration(dl_load)
            if tracker_result is not None:
                tracker_iter, _ = tracker_result
                if tracker_iter not in candidates:
                    candidates.append(tracker_iter)

            # Prefer the latest iteration that has a dataloader state file.
            candidates.sort(reverse=True)
            for cand_iter in candidates:
                data_save_name = get_checkpoint_name(
                    dl_load,
                    cand_iter,
                    pipeline_rank=0,
                    basename=f"train_dataloader_dprank{dp_rank:03d}.pt",
                )
                if os.path.exists(data_save_name):
                    try:
                        dataset_state_dict = torch.load(data_save_name, map_location="cpu", weights_only=False)
                        train_iter.load_state(dataset_state_dict["dataloader_state_dict"])
                        print_rank_0(
                            f"Restored dataloader state from {data_save_name} "
                            f"(step={dataset_state_dict['dataloader_state_dict'].get('step', '?')})"
                        )
                        restored = True
                        break
                    except Exception as e:
                        print_rank_0(f"WARNING: Failed to restore dataloader state from {data_save_name}: {e}")

            if not restored:
                print_rank_0("No dataloader state found to restore, starting from scratch")
    else:
        def _preprocess_iter(dl_iter):
            for batch in dl_iter:
                if preprocessor is not None:
                    yield preprocessor(batch)
                else:
                    yield batch

        train_iter = _preprocess_iter(_cyclic_iter(dataloader))

    return train_iter


def _build_cylic_iterator(
    dataset: Union["Dataset", "IterableDataset"],
    consumed_samples: int,
    data_collator: DataCollatorForSupervisedDataset,
):
    """build data iterator for sft"""
    if dataset is None:
        return None

    args = get_args()

    _dataloader_kwargs = {}
    if args.sft_data_streaming:
        # split distributed dataset for streaming
        dataset = split_dataset_by_node(
            dataset=dataset,
            rank=mpu.get_data_parallel_rank(),
            world_size=mpu.get_data_parallel_world_size(),
        )

        dataset = dataset.shuffle(
            buffer_size=args.streaming_buffer_size,
            seed=args.seed,
        )

        _dataloader_kwargs = dict(
            batch_size=args.micro_batch_size,
        )
    else:
        # build distribued sampler for non-streaming dataset
        if args.enable_chunkpipe:
            num_microbatches = args.global_batch_size // (
                args.micro_batch_size * mpu.get_data_parallel_world_size()
            )
            _batch_sampler = ChunkPipeGroupBatchSampler(
                dataset,
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=args.micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size(),
                num_microbatches=num_microbatches,
                seed=args.seed,
                enable_synthesis=getattr(args, "chunkpipe_enable_synthesis", False),
            )
        else:
            _batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,  # not support for streaming now!
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding,
        )

        _dataloader_kwargs = dict(
            batch_sampler=_batch_sampler,
            persistent_workers=True if args.num_workers > 0 else False,
        )

    dataloader = DataLoader(
        dataset,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        **_dataloader_kwargs,
    )

    if args.dataloader_save is not None:
        base_iter = SavableCyclicIterator(dataloader)
    else:
        base_iter = iter(_cyclic_iter(dataloader))

    if args.enable_chunkpipe and not args.sft_data_streaming:
        base_iter = _bind_chunkpipe_queue_iter(
            base_iter,
            _batch_sampler._step_g_queue,
            _batch_sampler._composite_queue,
        )

    return base_iter


def build_sft_cyclic_iterators(
    train_ds: Optional[Union["Dataset", "IterableDataset"]],
    valid_ds: Optional[Union["Dataset", "IterableDataset"]],
    test_ds: Optional[Union["Dataset", "IterableDataset"]],
    data_collator: Optional[DataCollatorForSupervisedDataset],
):
    """build data iterators for sft"""
    args = get_args()
    train_iter = _build_cylic_iterator(
        train_ds, args.consumed_train_samples, data_collator
    )
    valid_iter = _build_cylic_iterator(
        valid_ds, 0 if args.skip_train else args.consumed_valid_samples, data_collator
    )
    test_iter = _build_cylic_iterator(test_ds, 0, data_collator)
    return train_iter, valid_iter, test_iter


def _get_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """create attention mask"""
    args = get_args()
    current_device = attention_mask.device
    batch_size, seq_length = attention_mask.shape

    # Only used attn_mask when attn_mask_type in [padding, padding_causal, arbitrary] in TE
    # TODO: for multi-acceleator, maybe we should update attn_mask_type and attention_mask shape

    if args.context_parallel_size > 1:
        # Firstly, context parallel only support causal mask in TE now.
        # Secondly, when context-parallel is enabled, the input data is of a relatively long length,
        # and micro-batch-size does not need to be increased, nor padding occurs
        # create causal mask here, shape [B, 1, S, S].
        attention_mask = torch.tril(
            torch.ones(
                (batch_size, seq_length, seq_length),
                dtype=torch.long,
                device=current_device,
            )
        )
        attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()
    else:
        # create mask for te, shape [B, 1, 1, S]. attn_mask_type is padding_causal or causal.
        attention_mask.unsqueeze_(1).unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()

    return attention_mask


def _get_packed_sequence_params(attention_mask: torch.Tensor) -> PackedSeqParams:
    """create packed sequence params"""
    # assume micro_batch_size == 1
    assert attention_mask.shape[0] == 1, "attention_mask should be of shape [1, S]"

    packed_seq_params = PackedSeqParams()
    packed_seq_params.qkv_format = "thd"

    # calculate cu_seqlens_q
    # example: mask = [[1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 5, 0, 0]]
    # expacted cu_seqlens_q = [0, 2, 5, 7, 8, 11, 13]
    max_num = attention_mask.max().item()
    reduced_mask = torch.bincount(attention_mask.view(-1), minlength=max_num + 1)
    reduced_mask = reduced_mask[1:].to(dtype=torch.int32, device=attention_mask.device)

    cu_seqlens = reduced_mask.cumsum(dim=0).to(torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device=attention_mask.device)
    # The lengths of padding tokens must also be taken into account in cu_seqlens;
    # otherwise, the attention calculation will be incorrect.
    cu_seqlens[-1] = attention_mask.shape[1]
    cu_seqlens = torch.cat((zero, cu_seqlens))

    packed_seq_params.cu_seqlens_q = cu_seqlens
    packed_seq_params.cu_seqlens_kv = cu_seqlens  # just for self-attention
    packed_seq_params.max_seqlen_q = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    packed_seq_params.max_seqlen_kv = packed_seq_params.max_seqlen_q

    return packed_seq_params


