# AGENTS.md — LoongForge 训练框架

本文件是 Agent（Claude Code 等）理解项目结构的唯一信息源。

## 1. 目录地图

```text
loongforge/
├── train.py                          # 统一入口
├── training/
│   ├── registry.py                   # (method, engine, family) 注册
│   ├── mcore_runner.py               # MCore 训练主循环
│   ├── train_logging.py
│   └── methods/
│       ├── pretrain_llm.py  pretrain_vlm.py  pretrain_wan.py  pretrain_qwen_image.py
│       ├── sft_llm.py  sft_vlm.py  sft_internvl.py  sft_ernie.py
│       └── sft_embodied.py  sft_lingbot_va.py
├── engines/
│   ├── mcore/
│   │   ├── parser.py  arguments.py  validators.py  global_vars.py  constants.py
│   │   ├── initialize.py  model_setup.py  model_config.py
│   │   ├── train_step.py  checkpointing.py
│   │   ├── tokenizer/
│   │   ├── parallel/                 # context_parallel, batch_broadcast, chunkpipe …
│   │   └── optimizations/            # fine_grained, cuda_graph, fp8, offload …
│   └── torch/
│       ├── parser.py  arguments.py  validators.py  global_vars.py
│       ├── initialize.py  train_step.py  checkpointing.py
│       ├── lora.py  profiling.py
│       ├── distributed/  optimizer/
│       ├── trainers/                 # 旧 trainer（Phase 2B 迁移中）
│       └── optimizations/{groot_n1_6,groot_n1_7,lingbot_va}/
├── models/
│   ├── catalog.py                    # 模型名 → engine、配置路径、family
│   ├── mcore_registry.py             # MCore family → arch/config/provider
│   ├── language/                     # LLM backbone: qwen3, llama, deepseek …
│   ├── vision/                       # 视觉 encoder: qwen3_vl, internvl, moon …
│   ├── multimodal/                   # 跨模态组合模型
│   ├── diffusion/                    # wan, qwen_image
│   ├── embodied/                     # registry.py + pi05, groot, lingbot_va …
│   └── common/                       # 共用层、peft
├── data/
│   ├── sft_dataset.py  hf_dataset.py  sft_data_collator.py  sft_dataloader.py
│   ├── chat_template.py  chat_templates/
│   ├── multimodal/                   # task_encoder, packer, flavors, plugins
│   ├── video/
│   └── embodied/                     # datasets/, transforms/
└── evaluation/embodied/              # adapters, factories, metrics …
```

辅助目录：`configs/models/`（Hydra YAML）、`examples/`、`examples_xpu/`、`tools/`、`ops/`、`patches/`、`tests/`。

## 2. 训练调用链

```text
train.py
  → models.catalog 定位 engine、模型配置、模型族
  → engine parser + data.catalog 解析配置
  → training.registry 校验组合，选定 method、runner、执行优化
  → runner: engine.initialize → 组装 data 与 model → engine.model_setup
  → engine.checkpointing 恢复状态；data 恢复 iterator
  → 循环调用 engine.train_step(method 的 forward/loss, iterator)
  → runner 汇总日志，决定何时验证、保存、退出
```

**MCore 路径**：`train.py → catalog → mcore/parser → registry → mcore_runner → mcore/initialize → model_setup → mcore/train_step(method.forward_step) → checkpointing`

**Torch 路径**：`train.py → catalog → torch/parser → registry → torch_runner → torch/initialize → torch/train_step(method.forward/loss) → checkpointing`

## 3. 依赖规则

```text
train → catalog / parser / training.registry → runner
runner → method + data + models + engine
method → 模型计算接口 + 数据协议
engine → 框架执行 API + 传入的模型和数据对象
models → 模型库和层 API（可用 MCore/TE 的层）
data → 数据库和 processor
```

关键约束：
- **models** 不导入 training、runner、DataLoader，不读取 engine 全局训练状态。
- **data** 不依赖 engine context，不创建进程组。
- **engine** 不扫描 method，不按模型名复制训练循环。
- catalog 和配置在导入阶段不加载未选中的 engine、权重或可选依赖。

## 4. 扩展清单

| 新需求 | 必改 | 可复用 |
|---|---|---|
| 新 LLM/VLM | 模型包、`mcore_registry`、`catalog`、`configs/models/<family>/`、`examples/` | method、runner、engine |
| 新具身模型 | 模型包、`embodied/registry`、`catalog`、`transforms/<model>/`、`configs/`、`examples/` | datasets、dataloader、`sft_embodied`、torch runner |
| 新数据格式 | `data/embodied/datasets/`，必要时在 dataset_builder 注册 | 模型、method、engine |
| 新 loss/训练目标 | `training/methods/<phase>_<scope>.py` + `training/registry.py` | runner、train_step |
| 新通信/graph 优化 | `engines/<engine>/optimizations/` + 在 registry 登记 | method、data、runner |
| 新训练后端 | `engines/<new>/`，实现同名角色文件 + `<new>_runner.py` | method、模型、数据 |

## 5. 命名规则

### 目录名

| 名称 | 含义 |
|---|---|
| `language` | 语言 backbone 和 decoder（替换 `foundation`） |
| `vision` | 视觉 encoder 和 projector（替换 `encoder`） |
| `multimodal` | 跨模态组合（替换 `omni_models`） |
| `embodied` | 具身策略、动作模型、世界动作模型 |
| `diffusion` | 扩散模型 |
| `common` | 多领域共用组件 |
| `methods` | 训练目标与 forward/loss 组合 |

### 文件名

| 角色 | 命名 |
|---|---|
| 模型主实现 / 配置 / processor | `modeling_<family>.py`、`configuration_<family>.py`、`processing_<family>.py` |
| Layer spec | `<family>_layer_specs.py` |
| 模型构造 | `<scope>_model_provider.py`（如 `llm_model_provider.py`） |
| 训练方法 | `pretrain_<scope>.py`、`sft_<scope>.py` |
| engine 角色文件 | `initialize.py`、`train_step.py`、`checkpointing.py`、`parser.py`、`arguments.py`、`validators.py`、`global_vars.py` |
| runner | `mcore_runner.py`、`torch_runner.py` |

## 6. 验证命令

在仓库根目录执行：

```bash
set -o pipefail
git diff --check
.venv/bin/python -m compileall -q loongforge tools
find examples examples_xpu -type f -name '*.sh' -print0 | xargs -0 -r -n1 bash -n
.venv/bin/python -m pytest tests/test_train_dispatch.py tests/test_layout_paths.py tests/test_vlm_dataloader_read_order.py
rg -n 'examples/embodied|configs/models/embodied|loongforge[/.]models[/.](foundation|encoder|omni_models|custom|factory|dispatch)([/.]|$)|loongforge[/.]engines[/.](mcore[/.](pretrain|sft|diffusion|megatron_trainer|training_utils|trainer_builder|entrypoint)|torch[/.](trainers|entrypoint))([/.]|$)' \
  loongforge configs examples examples_xpu tools tests docs .github README* pyproject.toml
```

## 7. 禁止事项

1. **不新建 `utils/` 或 `custom/` 目录**：已有 `*_utils.py` 保留，但不作为新功能的默认落点。
2. **不 fork 上游函数**：Loong-Megatron 中已有的函数直接 import；需要扩展点时在上游加 hook。
3. **不在 `__init__.py` 中扫描导入全部模型**：选中的模块才导入，依赖缺失时报出具体模块名。
4. **models/data 中不调用 `get_args()`**（Phase 2C TODO）：模型和数据层不读取 engine 全局训练状态。
