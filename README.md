# Memory Dev

## Sweep Train (PKM Warmup)

基本启动方式：

```bash
CUDA_VISIBLE_DEVICES=0 python docs/run_t5_pkm_warmup_sweep.py --num_workers 1 --worker_id 0
```

多卡并行（每张卡跑不同的 sweep 组合）：

```bash
# GPU 0 跑 worker 0，GPU 1 跑 worker 1
CUDA_VISIBLE_DEVICES=0 python docs/run_t5_pkm_warmup_sweep.py --num_workers 2 --worker_id 0 &
CUDA_VISIBLE_DEVICES=1 python docs/run_t5_pkm_warmup_sweep.py --num_workers 2 --worker_id 1 &
```

常用参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dataset` | `Toys_and_Games` | 数据集名称 |
| `--data_root` | `./data` | 数据根目录 |
| `--epochs` | 读取 sweep yaml | 每个阶段训练 epoch 数 |
| `--num_workers` | `1` | 并行 worker 总数 |
| `--worker_id` | `0` | 当前 worker 编号（从 0 开始） |
| `--sweep_yaml` | `docs/pkm_warmup_sweep.yaml` | sweep 配置文件 |
| `--result_jsonl` | `log/{dataset}/.../result.jsonl` | 结果输出路径 |
| `--resume_from_result` | 无 | 跳过已完成的实验（断点续跑） |
| `--cleanup_ckpt` | `1` | 实验结束后是否删除 checkpoint |
| `--dry_run` | 无 | 只打印命令不实际运行 |

Sweep 的搜索空间在 `docs/pkm_warmup_sweep.yaml` 里定义，当前配置：

```yaml
sweep_params:
  d0_warmup_epochs: [0]          # D0 阶段 warmup epoch 候选
  finetune_warmup_epochs: [30, 40]  # D1~D3 阶段 warmup epoch 候选
```

结果汇总到 `log/{dataset}/sweep_t5_pkm_warmup/result.jsonl`，运行状态记录在同目录的 `sweep_status.jsonl`。
