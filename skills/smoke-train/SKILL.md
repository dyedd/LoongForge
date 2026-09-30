# smoke-train

最小训练冒烟测试：跑几步、保存检查点、恢复、对比 loss。

> **需要 GPU 环境**。无 GPU 时此 skill 不可执行。

## MCore 路径（Qwen3 pretrain）

```bash
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
  torchrun --nproc_per_node 1 $LOONGFORGE_PATH/loongforge/train.py \
    --model-name qwen3_8b --training-phase pretrain \
    --micro-batch-size 1 --global-batch-size 1 \
    --train-iters 2 --save-interval 2 \
    --save /tmp/smoke_mcore --load /tmp/smoke_mcore
# 恢复 1 步
# 将 --train-iters 改为 3，重新运行，确认从 step 2 恢复且 loss 连续
```

## Torch 路径（Pi05 SFT）

```bash
bash examples/pi05/finetune_pi05_ddp.sh  # 修改为 2 步
# 恢复 1 步，检查 loss 连续
```

## 检查项

1. 训练 2 步无报错，loss 非 NaN/Inf。
2. 检查点保存成功。
3. 从检查点恢复后 step 编号连续。
4. 恢复后首步 loss 与中断前最后一步的 loss 量级一致。
