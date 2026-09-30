# check-layout

验证目录结构、导入路径和旧引用是否已清理。

## 命令

```bash
set -euo pipefail

# 1. Python 编译检查
.venv/bin/python -m compileall -q loongforge tools

# 2. Shell 语法检查
find examples examples_xpu -type f -name '*.sh' -print0 | xargs -0 -r -n1 bash -n

# 3. 布局和调度测试
.venv/bin/python -m pytest tests/test_train_dispatch.py tests/test_layout_paths.py tests/test_vlm_dataloader_read_order.py

# 4. 旧路径扫描（应无输出）
rg -n 'examples/embodied|configs/models/embodied|loongforge[/.]models[/.](foundation|encoder|omni_models|custom|factory|dispatch)([/.]|$)|loongforge[/.]engines[/.](mcore[/.](pretrain|sft|diffusion|megatron_trainer|training_utils|trainer_builder|entrypoint)|torch[/.](trainers|entrypoint))([/.]|$)' \
  loongforge configs examples examples_xpu tools tests docs .github README* pyproject.toml

# 5. 导入目标检查
.venv/bin/python -m pytest tests/test_import_targets.py
```

任何步骤失败即表示有残留的旧路径或结构问题，需要修复后再继续。
