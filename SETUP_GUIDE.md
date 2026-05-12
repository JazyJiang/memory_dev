# Tiger Memory-Routing CL 实验复现指南

从零开始：数据下载 → 数据处理 → RQ-VAE 生成语义 ID → 跑 CL Baseline/Routing 实验。

---

## 0. 环境准备

```bash
# Python 3.10+, PyTorch 2.x, CUDA 12+
pip install torch transformers omegaconf hydra-core fire loguru tqdm scikit-learn numpy pandas

# RQ-VAE 需要
pip install sentence-transformers

# 可选 (加速/多卡)
pip install accelerate deepspeed
```

---

## 1. 数据下载

```bash
cd raw_data/

# Toys_and_Games (推荐先跑这个，数据量小)
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Toys_and_Games.json.gz
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Toys_and_Games.json.gz
gunzip Toys_and_Games.json.gz
gunzip meta_Toys_and_Games.json.gz

# 其他数据集（按需下载）：
# CDs_and_Vinyl, Video_Games, Books (URL 格式同上，替换类别名即可)
```

---

## 2. 数据处理

### 2.1 时间分段 + K-core 过滤

```bash
cd data/

# Toys_and_Games: 2016-10 到 2018-11, K=5 core
python 0_process.py --category Toys_and_Games --K 5 --st_year 2016 --st_month 10 --ed_year 2018 --ed_month 11

# 输出: data/D0/ ~ D4/ 目录，每个包含 Toys_and_Games_5_2016-10-2018-11.csv
# 输出: data/info/Toys_and_Games_5_2016-10-2018-11_combine_tdcb_maps.npy (item metadata)
```

### 2.2 生成测试分组

```bash
# 按用户在各 period 的活跃度分组 (用于 per-group evaluation)
python 1_generate_group.py --data_root ./data --dataset Toys_and_Games_5_2016-10-2018-11

# 生成 user-group 映射
python 2_generate_user_group_map.py --data_root ./data --dataset_name Toys_and_Games_5_2016-10-2018-11 --n_groups 5
```

输出: `data/D{1,2,3,4}/groups/*.csv` (每 group 一个 test 文件)

---

## 3. RQ-VAE 生成语义 Item ID

```bash
cd RQ-VAE/

# Step 1: 用 T5 encoder 生成 item embedding
python generate_embeddings.py \
    --tdcb_path ../data/info/Toys_and_Games_5_2016-10-2018-11_combine_tdcb_maps.npy \
    --output_path ../data/info/Toys_and_Games.emb-t5-tdcb.npy \
    --model_name google-t5/t5-small \
    --batch_size 128 --device cuda:0

# Step 2: 训练 RQ-VAE + 生成离散 index
python generate_indices_sk.py \
    --embedding_path ../data/info/Toys_and_Games.emb-t5-tdcb.npy \
    --tdcb_path ../data/info/Toys_and_Games_5_2016-10-2018-11_combine_tdcb_maps.npy \
    --output_path ../data/info/Toys_and_Games.TIGER-index.json \
    --num_codebooks 3 --codebook_size 256 \
    --epochs 100 --batch_size 256 --device cuda:0
```

输出: `data/info/Toys_and_Games.TIGER-index.json` (每个 item 的语义 ID)

---

## 4. 跑 CL 实验

### 方式 A: 一键 Sweep（推荐）

`docs/run_delta_set_sweep.py` 自动完成 D0 train → D1-D4 test → D1 finetune → D2-D4 test → ... 全链路。

```bash
# 跑所有预定义的实验组合（baselines + routing + aux）
CUDA_VISIBLE_DEVICES=0 python docs/run_delta_set_sweep.py \
    --num_workers 1 --worker_id 0

# 多 GPU 并行（每张卡跑不同实验）
CUDA_VISIBLE_DEVICES=0 python docs/run_delta_set_sweep.py --num_workers 4 --worker_id 0 &
CUDA_VISIBLE_DEVICES=1 python docs/run_delta_set_sweep.py --num_workers 4 --worker_id 1 &
CUDA_VISIBLE_DEVICES=2 python docs/run_delta_set_sweep.py --num_workers 4 --worker_id 2 &
CUDA_VISIBLE_DEVICES=3 python docs/run_delta_set_sweep.py --num_workers 4 --worker_id 3 &
```

**Sweep 内的实验组合**（在 `_DELTA_COMBOS` 列表中定义）：

| Label | 说明 |
|-------|------|
| `h2_t5` | Baseline: history=2, T5 |
| `h10_t5` | Baseline: history=10, T5 |
| `h10_pkm` | T5 + PKM (decoder layer 2) |
| `h10_route_ffn` | T5 + Cross-Attention Routing |
| `h10_route_ffn_aux` | T5 + Routing + Aux Loss |
| `h10_route_pkm_gate_aux` | Full combo: Routing + PKM + Gate + Aux |

结果输出到 `log/Toys_and_Games/sweep_delta_set/result.jsonl`。

### 方式 B: 单独跑 shell 脚本

```bash
# 修改 slurm/train_t5.sh 中的参数后直接运行
bash slurm/train_t5.sh
```

主要参数：
- `DATASET`: 数据集名
- `T5_PK_IS_ENABLED`: 是否启用 PKM (0/1)
- `LR`: 学习率 (推荐 3e-4)
- `EPOCHS`: 每 period 训练 epoch 数 (推荐 50)
- `BATCH_SIZE`: batch size (推荐 256-512)

### 方式 C: 直接调用 train.py / test.py

```bash
# 训练
torchrun --nproc_per_node=1 train.py \
    config=configs/train_t5_pkm_warmup.yaml \
    strategy=t5_seq2seq \
    model.t5_seq2seq.base_model=google-t5/t5-small \
    train.output_dir=./ckpt/Toys_and_Games/my_exp \
    dataset.data_path=./data \
    dataset.name=Toys_and_Games \
    dataset.train_file=./data/D0/Toys_and_Games_5_2016-10-2018-11.csv \
    dataset.index_file=.TIGER-index.json \
    train.learning_rate=3e-4 \
    train.epochs=50 \
    train.batch_size=512

# 测试
python test.py \
    config=configs/train_t5_pkm_warmup.yaml \
    model.type=t5_seq2seq \
    model.ckpt_path=./ckpt/Toys_and_Games/my_exp \
    model.base_model=google-t5/t5-small \
    dataset.name=Toys_and_Games \
    dataset.data_path=./data \
    dataset.test_file=./data/D1/groups/group_0.csv \
    dataset.index_file=.TIGER-index.json \
    test.batch_size=256 \
    test.num_beams=20
```

---

## 5. 只跑 Baseline（给合作者）

如果只需要跑 **不带 routing/PKM 的纯 T5 baseline**：

```bash
# 单卡跑 h=2 和 h=10 两个 baseline
CUDA_VISIBLE_DEVICES=0 python docs/run_delta_set_sweep.py \
    --num_workers 1 --worker_id 0 \
    --labels h2_t5,h10_t5
```

或手动指定 `--filter_labels` 只跑你需要的实验。

用 --labels 参数指定要跑的实验 label（逗号分隔），不指定则跑全部。

---

## 6. 结果汇总

```bash
# 从 test log 中提取 metrics 并画图
python plot_test_results.py \
    --log_dir ./log/Toys_and_Games/test \
    --out_dir ./log/Toys_and_Games/plots \
    --dataset Toys_and_Games
```

Metrics 包含: Recall@5/10/20, NDCG@5/10/20, per user-group 分组。

---

## 目录结构总览

```
memory_dev/
├── train.py              # 主训练入口
├── test.py               # 评估入口 (beam search + top-N)
├── data.py               # Dataset 类
├── configs/              # OmegaConf YAML 配置
├── models/
│   ├── routed_t5.py      # Cross-attention routing + aux head
│   └── decoder_only/     # 自定义 decoder-only Transformer
├── pkm/                  # Product-Key Memory 模块
├── RQ-VAE/               # 语义 ID 生成
├── data/
│   ├── 0_process.py      # 数据处理: 时间分段 + K-core
│   ├── 1_generate_group.py   # 测试分组
│   ├── 2_generate_user_group_map.py
│   ├── info/             # TIGER index + embedding
│   └── D0/ ~ D4/        # 各 period 数据
├── raw_data/             # 原始 Amazon review 数据
├── docs/
│   └── run_delta_set_sweep.py  # 一键 sweep runner
├── slurm/                # 训练脚本模板
└── log/                  # 训练/测试日志 + 结果
```

---

## 常见问题

**Q: T5-small 模型会自动下载吗？**
A: 是的，第一次运行时 HuggingFace 会自动下载 `google-t5/t5-small`。如果网络受限，提前下载放到本地路径，然后修改 `BASE_MODEL` 指向本地目录。

**Q: 一个 baseline 实验大概跑多久？**
A: Toys_and_Games (T5-small, 50 epochs/period, 4 periods, 单卡 V100/A100): ~2-3h 全部跑完。

**Q: 如何只跑特定数据集？**
A: `run_delta_set_sweep.py` 默认跑 Toys_and_Games。改 `--dataset` 参数即可切换，但需确保对应的数据已处理完成。

**Q: checkpoint 太大怎么办？**
A: sweep runner 默认训完一个实验就删 checkpoint (`--cleanup_ckpt 1`)。只保留 log 和结果 JSONL。
