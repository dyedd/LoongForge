# add-model

添加新 LLM/VLM 或具身模型的检查清单。

## LLM / VLM（MCore 路径）

1. 在 `loongforge/models/language/<family>/` 或 `vision/<family>/` 创建模型包：
   - `configuration_<family>.py` — config dataclass，用 `@register_model_config(family, arch)` 注册
   - `modeling_<family>.py` — transformer spec
   - `<family>_layer_specs.py` — layer spec（如需）
2. 在 `loongforge/models/mcore_registry.py` 注册 family → arch/config/provider。
3. 在 `loongforge/models/catalog.py` 添加条目：模型名 → engine、YAML 路径、family。
4. 在 `configs/models/<family>/` 添加 Hydra YAML。
5. 如需新训练方法，在 `loongforge/training/methods/` 添加并在 `training/registry.py` 注册。
6. 在 `examples/<family>/` 添加启动脚本（pretrain/finetuning/checkpoint_convert）。

## 具身模型（Torch 路径）

1. 在 `loongforge/models/embodied/<model>/` 创建模型包。
2. 在 `loongforge/models/embodied/registry.py` 注册 model type → 构造入口。
3. 在 `loongforge/models/catalog.py` 添加条目（engine=torch）。
4. 在 `loongforge/data/embodied/transforms/<model>/` 添加数据变换（如需）。
5. 在 `configs/models/<family>/` 添加 YAML。
6. 在 `examples/<family>/` 添加启动脚本。

## 验证

```bash
.venv/bin/python -m compileall -q loongforge/models/<family>
.venv/bin/python -c "from loongforge.models.catalog import MODEL_CONFIG_REGISTRY; assert '<model_name>' in MODEL_CONFIG_REGISTRY"
.venv/bin/python -m pytest tests/test_train_dispatch.py -k <family>
```
