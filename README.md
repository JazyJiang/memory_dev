# Tiger-CL: Continual Learning for Sequential Recommendation

T5-based generative sequential recommender with **Product-Key Memory (PKM)**, **Cross-Attention Routing**, and **Auxiliary Loss** for continual learning across temporal data splits.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data (download + process + generate TIGER index)
bash scripts/setup_data.sh Toys_and_Games cuda:0

# 3. Run experiments
bash run.sh Toys_and_Games baseline_h10 0      # single baseline
bash run.sh Toys_and_Games all 0               # all methods
```

---

## 1. Data Preparation

### 1.1 Supported Datasets

数据来源: [Amazon Review 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)

| Dataset | Time Range | K-core | Approx. Items |
|---------|-----------|--------|---------------|
| `Toys_and_Games` | 2016-10 ~ 2018-11 | 5 | ~12K |
| `Video_Games` | 2012-10 ~ 2018-11 | 5 | ~17K |
| `CDs_and_Vinyl` | 2014-10 ~ 2018-11 | 5 | ~35K |
| `Books` | 2017-10 ~ 2018-11 | 5 | ~142K |

每个数据集按时间等分为 5 个 period (D0–D4)，用于 continual learning 评估。

### 1.2 数据处理 Pipeline

`scripts/setup_data.sh` 一键完成：

```bash
bash scripts/setup_data.sh <dataset> [device]
# Example: bash scripts/setup_data.sh Toys_and_Games cuda:0
```

内部 7 步：

| Step | 脚本 | 说明 |
|------|------|------|
| 1 | `wget` | 下载 Amazon review + metadata JSON |
| 2 | `data/0_process.py` | 时间分段 + K-core=5 过滤，输出 D0–D4 CSV |
| 3 | `data/1_generate_group.py` | 按用户活跃度分 5 组 (G0–G4)，用于 per-group evaluation |
| 4 | `data/2_generate_user_group_map.py` | 生成 user→group 映射 |
| 5 | `transformers` | 自动下载 `google-t5/t5-small` 模型 (≈240MB) |
| 6 | `RQ-VAE/generate_embeddings.py` | 用 T5 encoder (mean pooling) 生成 item embedding |
| 7 | `RQ-VAE/main.py` | 训练 RQ-VAE，生成 TIGER semantic item ID |

### 1.3 RQ-VAE 参数

RQ-VAE 将 T5 embedding 离散化为 4 级 codebook ID (`<a_X><b_X><c_X><d_X>`)：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_emb_list` | `256 256 256 256` | 每级 codebook 大小 |
| `--e_dim` | 32 | codebook embedding 维度 |
| `--sk_epsilons` | `0.0 0.0 0.0 0.003` | Sinkhorn 正则（最后一级去冲突） |
| `--epochs` | 5000 | 训练 epoch |
| `--batch_size` | 480 | batch size |
| `--lr` | 1e-3 | 学习率 |
| `--layers` | `2048 1024 512 256 128 64` | encoder MLP 层 |

输出: `data/info/<dataset>.TIGER-index.json`

---

## 2. Training

### 2.1 Base Model

- **Model**: `google-t5/t5-small` (60M params)
- **Config**: `configs/train_t5_pkm_warmup.yaml`

### 2.2 默认训练参数

| 参数 | D0 (from scratch) | D1+ (finetune) |
|------|-------------------|----------------|
| Learning rate | 1e-3 | 1e-3 (可调 `--ft_lr`) |
| Epochs | 50 | 50 (可调 `--ft_epochs`) |
| Batch size | 512 | 512 |
| Optimizer | AdamW | AdamW |
| LR scheduler | cosine | cosine |
| Warmup ratio | 0.01 | 0.01 |
| Weight decay | 0.001 | 0.001 |
| Max sequence length | 512 | 512 |

### 2.3 Evaluation 参数

| 参数 | 默认值 |
|------|--------|
| Beam search width | 20 |
| Test batch size | 8 |
| Metrics | Recall@{5,10,20}, NDCG@{5,10,20} |
| Grouping | G0–G4 (按用户在 D_{t-1} 的活跃度分组，评估 D_t) |

---

## 3. Experiment Methods

### 3.1 一键运行

```bash
bash run.sh <dataset> <method> <gpu_id>
```

| Method | History | PKM | Routing | Aux Loss | Sweep Label |
|--------|---------|-----|---------|----------|-------------|
| `baseline_h2` | 2 | - | - | - | `h2_t5` |
| `baseline_h10` | 10 | - | - | - | `h10_t5` |
| `pkm` | 10 | decoder layer 3 | - | - | `h10_pkm` |
| `routing` | 10 | - | early_layers=[3] | - | `h10_route_ffn` |
| `routing_aux` | 10 | - | early_layers=[3] | weight=0.1 | `h10_route_ffn_aux` |
| `pkm_routing_aux` | 10 | decoder layer 3, gated | early_layers=[3] | weight=0.1 | `h10_route_pkm_gate_aux` |
| `all` | - | - | - | - | 跑以上全部 |

### 3.2 Method Details

#### Baseline (T5 only)

纯 T5-small seq2seq，history 截断到 h 条：

```yaml
# No extra config needed
dataset.max_his_len: 10   # or 2
pkm.t5_seq2seq.pk_is_enabled: false
routing.enabled: false
```

#### Product-Key Memory (PKM)

在 T5 decoder 指定层插入 PKM 模块，扩展模型记忆容量：

```yaml
pkm.t5_seq2seq:
  pk_is_enabled: true
  pk_decoder_layers: [3]       # 哪些 decoder layer 插入 PKM
  pk_mem_n_keys: 128           # key 数量 (per head)
  pk_mem_heads: 4              # memory heads
  pk_mem_knn: 32               # top-k nearest keys
  pk_topk: 8                   # final top-k
  pk_mem_k_dim: 512            # key embedding dim
  pk_mem_gated: false          # gated residual (true for full combo)
  pk_warmup_epochs: 10         # warmup: 先冻结 PKM 训 backbone
```

#### Cross-Attention Routing

decoder 的不同层 attend 到不同的 history：early layers attend 全部 history，其余层只 attend 最近 `recent_history_len` 条：

```yaml
routing:
  enabled: true
  early_layers: [3]            # 哪些 decoder layer attend 全部 history
  recent_history_len: 2        # 其余层只看最近几条
  gate_l1_weight: 0.0          # gate sparsity loss (0.01 for gated variant)
```

#### Auxiliary Prediction Loss

在 early layer 输出加一个 prediction head，让 early layer 也学到 item 预测信号：

```yaml
routing:
  aux_loss_weight: 0.1         # aux loss 权重 (0 = 关闭)
  early_layers: [3]            # aux head 加在哪些层
```

Note: `aux_loss_weight > 0` 即启用，不要求 `routing.enabled=true`。可单独用。

### 3.3 高级用法: 直接调用 sweep runner

```bash
# 跑指定 label 的实验
CUDA_VISIBLE_DEVICES=0 python docs/run_delta_set_sweep.py \
    --dataset Toys_and_Games \
    --time_range 2016-10-2018-11 \
    --labels h10_t5,h10_route_ffn_aux \
    --epochs 50 --lr 1e-3 --batch_size 512 \
    --ft_lr 3e-4 --ft_epochs 20

# 多 GPU 并行 (按 worker 分配实验)
CUDA_VISIBLE_DEVICES=0 python docs/run_delta_set_sweep.py --num_workers 4 --worker_id 0 &
CUDA_VISIBLE_DEVICES=1 python docs/run_delta_set_sweep.py --num_workers 4 --worker_id 1 &
CUDA_VISIBLE_DEVICES=2 python docs/run_delta_set_sweep.py --num_workers 4 --worker_id 2 &
CUDA_VISIBLE_DEVICES=3 python docs/run_delta_set_sweep.py --num_workers 4 --worker_id 3 &
```

---

## 4. CL Protocol

每个实验自动跑完整 continual learning 链：

```
D0 train (from scratch) → eval on D1 (per group G0-G4)
D1 finetune (from D0 ckpt) → eval on D2
D2 finetune (from D1 ckpt) → eval on D3
D3 finetune (from D2 ckpt) → eval on D4
```

结果输出到 `log/<dataset>/delta_set_sweep/result.jsonl`，每行一个实验的完整 metrics。

---

## 5. Project Structure

```
├── run.sh                        # 统一实验启动器
├── scripts/
│   └── setup_data.sh             # 数据下载 + 处理 + RQ-VAE pipeline
├── train.py                      # 训练入口 (torchrun)
├── test.py                       # 评估入口 (beam search + ranking metrics)
├── data.py                       # Dataset 类
├── collator.py                   # Data collator (seq2seq padding)
├── configs/
│   └── train_t5_pkm_warmup.yaml  # 默认训练配置
├── models/
│   └── routed_t5.py              # Cross-attention routing + aux prediction head
├── pkm/
│   └── memory.py                 # Product-Key Memory 模块
├── RQ-VAE/
│   ├── main.py                   # RQ-VAE 训练
│   ├── generate_embeddings.py    # T5 item embedding 生成
│   └── generate_indices_sk.py    # Sinkhorn 去冲突 + 生成 TIGER index
├── data/
│   ├── 0_process.py              # 时间分段 + K-core 过滤
│   ├── 1_generate_group.py       # 测试集用户分组
│   └── 2_generate_user_group_map.py
├── docs/
│   └── run_delta_set_sweep.py    # CL sweep runner (run.sh 调用)
└── raw_data/
    └── download.sh               # 数据集下载 (也集成在 setup_data.sh 中)
```
