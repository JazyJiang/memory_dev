# TODO

在论文风格的 decoder-only Transformer 中，将指定层的 FFN 替换为 `HashingMemory (PKM/PEER)`，其余 Attention/残差/Norm 结构保持不变，训练的输入输出格式与 loss 定义保持不变。

## Phase 1: Baseline + PKM 最小版本 + 训练稳定

- 落目录与入口
  - `models/decoder_only/`：模型实现
  - `train_decoder_only.py`：训练入口（或改造现有 `finetune.py` 指向新模型）
- 定义配置
  - `ModelArgs`：`d_model/n_layers/n_heads/ffn_dim/vocab_size/max_seq_len/dropout`
  - `ProductKeyArgs`：`is_enabled/pk_layers/mem_n_keys/topk/mem_dim/use_gating`
- 实现 baseline
  - `TransformerBlock`：`Attention (causal + RoPE + KV cache)` + `FFN` + `RMSNorm` + residual
  - `DecoderOnlyTransformer`：embedding + blocks + lm_head
  - `forward(input_ids, attention_mask=None, labels=None) -> {loss, logits}`（`labels=-100` 语义不变）
- 实现 PKM（torch 后端）
  - `pkm/memory.py`：`HashingMemory(hidden_states[B,T,D]) -> [B,T,D]`
  - keys/values 初始化；product-key 检索（两次 topk + 组合 + 再 topk）；读出（torch `embedding_bag` 或等价加权聚合）；输出投影回 `d_model`
- 集成与替换
  - `layer_id in pk_layers` 时以 `HashingMemory` 替换 baseline FFN
  - `pk_layers=[]` 等价 baseline
- 稳定与诊断
  - memory 参数组：`lr/weight_decay`（如需要）
  - 权重计算局部 fp32（如需要），输出 cast 回原 dtype
  - 记录：权重熵、topk 索引分布、memory 输出幅度

交付物：
- 小数据 forward/backward 通过，loss 正常下降
- 开启 PKM 后可训练且无 NaN/Inf，多 seed 行为稳定

## Phase 2: 性能版本（可回退）

- Triton embedding_bag
  - 接入 `github_ref/lingua/product_key/xformer_embeddingbag.py`
  - 后端开关：`backend=torch|triton`，torch 永久可用
- values 列切分
  - 接入 `github_ref/lingua/product_key/colwise_embedding_bag.py` 的列切分策略
  - 仅并行化 `values`，检索逻辑不变

交付物：
- 显存下降或吞吐提升达到目标；可一键回退 torch 后端

## Phase 3: 分布式与并行策略（按需）

- `mp_parallelize(mesh)` 接口
  - 对 memory values 做 memory-parallel
- 与 DDP/FSDP/TP 的组合验证
  - 单机多卡 -> 多机多卡（如需要）

交付物：
- 分布式训练收敛正常；并行策略行为可解释、可复现