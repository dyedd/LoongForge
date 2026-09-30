# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unified training registry.

Replaces engines/mcore/trainer_builder.py (decorator-based registration) and
engines/torch/trainers/trainer_builder.py (class lookup) with a single,
statically-declared registry.  All heavy imports (torch, megatron, method
modules) are deferred to call time via importlib.import_module.

Usage (MCore):
    method = resolve_method("qwen2_vl", "pretrain")
    # method is the lazily-imported module (e.g. pretrain_vlm)

Usage (Torch):
    trainer_cls = resolve_torch_trainer("GrootN1d6Trainer")
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Family -> group mapping.
# Built from engines/mcore/constants.py families.  Each concrete model family
# maps to a "group" that selects the method module.  Families that need their
# own method (intern_vl, ernie4_5_vl, wan, qwen_image) map to themselves.
# ---------------------------------------------------------------------------

_LLM_FAMILIES = (
    "llama", "llama2", "llama3", "llama3.1",
    "qwen", "qwen1.5", "qwen2", "qwen2.5", "qwen3", "qwen3_next",
    "kimi_k3_llm",
    "deepseek", "deepseek_v4",
    "internlm2.5",
    "minimax", "mimo", "glm",
)

_VLM_FAMILIES = (
    "qwen2_vl", "qwen2_5_vl", "qwen3_vl",
    "llava_ov_1_5", "vlm",
    "qwen3_5",
    "minicpm_v_4_6",
    "kimi_k2_5", "kimi_k2_6", "kimi_k3",
)

FAMILY_GROUP: Dict[str, str] = {}
for _f in _LLM_FAMILIES:
    FAMILY_GROUP[_f] = "llm"
for _f in _VLM_FAMILIES:
    FAMILY_GROUP[_f] = "vlm"
# Families with dedicated methods keep their own name as group key.
FAMILY_GROUP["intern_vl"] = "intern_vl"
FAMILY_GROUP["ernie4_5_vl"] = "ernie4_5_vl"
FAMILY_GROUP["wan2_1_i2v"] = "wan"
FAMILY_GROUP["wan2_2_i2v"] = "wan"
FAMILY_GROUP["qwen_image"] = "qwen_image"

# ---------------------------------------------------------------------------
# (group, phase) -> method module path   (MCore engine)
# ---------------------------------------------------------------------------

METHOD_REGISTRY: Dict[Tuple[str, str], str] = {
    # LLM
    ("llm", "pretrain"): "loongforge.training.methods.pretrain_llm",
    ("llm", "sft"):      "loongforge.training.methods.sft_llm",
    # VLM  (sft_vlm delegates to pretrain_vlm callbacks)
    ("vlm", "pretrain"): "loongforge.training.methods.pretrain_vlm",
    ("vlm", "sft"):      "loongforge.training.methods.sft_vlm",
    # Dedicated overrides
    ("intern_vl", "sft"):     "loongforge.training.methods.sft_internvl",
    ("ernie4_5_vl", "sft"):   "loongforge.training.methods.sft_ernie",
    ("wan", "pretrain"):      "loongforge.training.methods.pretrain_wan",
    ("qwen_image", "pretrain"): "loongforge.training.methods.pretrain_qwen_image",
}


def resolve_method(model_family: str, training_phase: str):
    """Lazily import and return the method module for *model_family* + *phase*.

    Raises ValueError for unsupported combinations.
    """
    group = FAMILY_GROUP.get(model_family, model_family)
    key = (group, training_phase)
    module_path = METHOD_REGISTRY.get(key)
    if module_path is None:
        raise ValueError(
            f"No method registered for ({model_family!r}, {training_phase!r}) "
            f"[resolved group={group!r}]. "
            f"Available: {sorted(METHOD_REGISTRY.keys())}"
        )
    return importlib.import_module(module_path)


# ---------------------------------------------------------------------------
# Torch engine: --trainer-type -> (module_path, class_name)
# Mirrors engines/torch/trainers/trainer_builder.py._TRAINER_CLASSES without
# importing torch at module level.
# ---------------------------------------------------------------------------

TORCH_TRAINER_REGISTRY: Dict[str, Tuple[str, str]] = {
    "FinetuneTrainer": (
        "loongforge.engines.torch.trainers.supervised.finetune_trainer",
        "FinetuneTrainer",
    ),
    "GrootN1d6Trainer": (
        "loongforge.engines.torch.trainers.custom.groot_n1_6",
        "GrootN1d6Trainer",
    ),
    "LingBotFinetuneTrainer": (
        "loongforge.engines.torch.trainers.custom.lingbot_va",
        "LingBotFinetuneTrainer",
    ),
    "GrootN1d7Trainer": (
        "loongforge.engines.torch.trainers.custom.groot_n1_7",
        "GrootN1d7Trainer",
    ),
}


def resolve_torch_trainer(trainer_type: str):
    """Lazily import and return the Torch trainer *class*.

    Raises ValueError for unknown --trainer-type values.
    """
    entry = TORCH_TRAINER_REGISTRY.get(trainer_type)
    if entry is None:
        raise ValueError(
            f"Unknown --trainer-type {trainer_type!r}. "
            f"Available: {sorted(TORCH_TRAINER_REGISTRY.keys())}"
        )
    module_path, class_name = entry
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)
