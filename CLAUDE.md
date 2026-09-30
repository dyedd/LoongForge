# CLAUDE.md

@AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LoongForge is large-scale transformer training framework built on top of Megatron-LM (as a patched fork: Loong-Megatron) and TransformerEngine. It supports LLMs, VLMs (Vision-Language Models), VLAs (Vision-Language-Action Models), and Diffusion Models across both NVIDIA GPUs and Kunlun XPUs. Training phases supported: pretrain and SFT (supervised fine-tuning).

## Build & Setup

### Quick Start (Docker — recommended)
```bash
git clone --recurse-submodules https://github.com/baidu-baige/LoongForge.git
# COMPILE_ENV: ampere | hopper | blackwell
docker build --build-arg COMPILE_ENV=hopper --build-arg ENABLE_LEROBOT=false \
  -t loongforge:latest -f ./LoongForge/docker/Dockerfile .
```

### Source Install
```bash
# 1. Clone with Megatron submodule
git clone --recurse-submodules https://github.com/baidu-baige/LoongForge.git
cd LoongForge

# 2. Install LoongForge + dependencies
uv pip install -e ".[gpu]"    # NVIDIA GPU
uv pip install -e ".[xpu]"    # Kunlun XPU

# 3. Setup TransformerEngine (clone, patch, build)
python setup_env.py --te-tag v2.9
```

Note: `setup_env.py` only handles TransformerEngine. Megatron-LM (Loong-Megatron) is a git submodule at `third_party/Loong-Megatron`, initialized via `--recurse-submodules`.

### Build Package
```bash
python -m build --sdist --wheel --outdir dist/
```

## Running Tests

E2E tests use a custom YAML-driven framework (`tests/llm_vlm/main.py`), not pytest.

The entry script does NOT download artifacts. Provision the datasets, HuggingFace base
models, and pre-converted checkpoints referenced by the selected configs first, then run:

```bash
# Run the default CI suite (all models in tests/llm_vlm/configs/)
bash tests/llm_vlm/main_start.sh
```

### Running a Single Model Test

Edit variables in `tests/llm_vlm/main_start.sh`:
```bash
# Run one model from tests/llm_vlm/configs/
model_names="qwen3_14b"

# Run one model from tests/llm_vlm/optional_configs/
model_names="deepseek_v2/deepseek_v2_lite"
include_optional=true

# Run an entire model series from optional_configs/
model_names="NONE"
optional_subdir="internvl2.5"
include_optional=true
```

Test configs: `tests/llm_vlm/configs/` (CI suite) and `tests/llm_vlm/optional_configs/` (regression, organized by model family). Each YAML defines model params and multi-step `scenarios` (checkpoint conversion + training).

### Embodied/VLA Regression Tests

`tests/embodied/` is the end-to-end regression suite for training scripts under
`loongforge/engines/torch/`, `loongforge/models/embodied/`, and `examples/`. Its entry point is
`tests/embodied/run.sh`; execution, metric parsing, and baseline comparison are owned by
`tests/embodied/cli.py`. Regression targets are registered in
`tests/embodied/config/scripts.yaml` and run serially in manifest order.

```bash
# List available embodied regression targets
bash tests/embodied/run.sh --list_models

# Run the full regression suite on a chip
bash tests/embodied/run.sh --chip a

# Run selected targets
bash tests/embodied/run.sh --chip a --models fastwam_ddp fastwam_ddp_zero1

# Collect baselines for the current chip
bash tests/embodied/run.sh --chip a --auto_collect_baseline

# Artifacts are provisioned by the CI workflow/self-hosted runner before this step.

# Validate commands/configuration without training
bash tests/embodied/run.sh --chip a --dry_run
```

Embodied test conventions:

- `tests/embodied/config/env.sh` centralizes `EMBODIED_CI_ROOT`,
  `LOCAL_VLA_ARTIFACTS_ROOT`, log, and baseline paths. Prefer environment
  overrides or this file when moving the suite to another machine.
- Add every new training script to `tests/embodied/config/scripts.yaml`; the manifest
  path is relative to `examples/`. Add a baseline under
  `tests/embodied/baseline/<chip>/<name>.json` for each supported chip.
- The executor injects `OUTPUT_DIR`, `TENSORBOARD_DIR`, and model-specific environment
  variables. Training scripts should expose environment overrides for data, checkpoints,
  caches, and output paths instead of relying on the executor to rewrite training args.
- Loss and `grad_norm` are hard-checked by default. Missing baselines, non-zero training
  exits, insufficient metrics, NaN/Inf, or skipped iterations fail the regression.
  Performance regressions warn by default; clear performance improvements may update the
  baseline.
- `--auto_collect_baseline` collects results without comparison. Baselines are chip-specific
  and must not be reused across different hardware without validation.
- For changes to video/action preprocessing, text-embedding caches, or distributed strategy,
  run the affected target's `--dry_run` first and then the real regression when artifacts
  and hardware are available.

## Training Launch Pattern

Training scripts use `torchrun` for distributed execution. The PYTHONPATH must include both Megatron-LM and LoongForge:

```bash
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun --nproc_per_node 8 --nnodes $NNODES ... \
    $LOONGFORGE_PATH/loongforge/train.py \
    --model-name <model-name> \
    --training-phase pretrain|sft \
    ...
```

Key arguments: `--model-name` (maps to an engine and config via `loongforge/models/catalog.py`) or `--config-file` (direct YAML path), `--training-phase` (pretrain/sft).

## Custom Ops: `ops/`

Custom CUDA kernels: `sparse_mla_fwd/`, `sparse_mla_bwd/` (sparse MLA attention), `lightning_indexer_bwd/`.

### Examples: `examples/`

Shell scripts for each supported model family with pretrain/SFT/checkpoint-conversion configs. Pattern: `examples/<model>/{pretrain,sft,checkpoint_convert}/`.

### XPU Support: `examples_xpu/`

Kunlun XPU training scripts, mirroring `examples/` structure.

## Patches

`patches/TransformerEngine_v2.9/` contains patch files applied to upstream TransformerEngine during setup. These implement LoongForge-specific optimizations and fixes.

## Code Review

When asked to review a pull request or a diff (e.g. via `@claude review this PR`), follow `skills/code-review/SKILL.md` exactly: its checklist, severity tags (🔴 Critical / 🟠 Major / 🟡 Minor / 🟢 Nit), and output format are the authoritative contract for review output.

**Posting findings — prefer inline comments.** For every finding tied to specific code, call the `mcp__github_inline_comment__create_inline_comment` tool to post it on the exact `file:line` (or `startLine`–`line` range) instead of listing it in the summary. Use the top-level summary comment ONLY for: the overall `Verdict`, the 2–4 sentence `Summary`, the `Tests` line, and the `Checklist` table. Each inline comment body should follow the same severity-tag format as the SKILL spec (`🔴 Critical: ...`, `🟠 Major: ...`, etc.). Set `confirmed: true` only on real review findings — never on probes or self-tests.
