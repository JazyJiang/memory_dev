```mermaid
flowchart TD
  MHA["mha<br>in: (B, T, D)<br>out: (B, T, D)"]
  OUT["output<br>shape: (B, T, D)"]

  MHA --> IN0

  subgraph PKM["memory layer: HashingMemory (PKM)"]
    direction TB

    DIMS["dims<br>B=batch<br>T=seq_len<br>D=d_model<br>bs=B*T<br>H=heads<br>K=k_dim<br>N=n_keys<br>KNN=knn<br>v_dim=value_dim"]

    IN0["h (input to PKM)<br>shape: (B, T, D)"] --> FLAT["reshape/view<br>(B, T, D) -> (bs, D)<br>bs = B*T"]
    FLAT --> IDROP["dropout(input)<br>shape: (bs, D)"]

    IDROP --> QPROJ["query_proj (QueryMLP)<br>Linear: (bs, D) -> (bs, H*K)<br>view: (bs, H*K) -> (bs*H, K)"]
    QPROJ --> QDROP["dropout(query)<br>shape: (bs*H, K)"]

    QDROP --> QRESH["get_indices: view query<br>(bs*H, K) -> (bs, H, K)"]
    QRESH --> SPLIT["split halves along K<br>q1,q2: (bs, H, K/2)"]

    KEYS["keys param (flattened storage)<br>param: (2*H*N, K/2)<br>view -> (H, 2, N, K/2)<br>keys1,keys2: (H, N, K/2)"]

    SPLIT --> DOT1["scores1 = headwise dot(q1, keys1)<br>(bs, H, K/2) x (H, N, K/2) -> (bs, H, N)"]
    SPLIT --> DOT2["scores2 = headwise dot(q2, keys2)<br>(bs, H, K/2) x (H, N, K/2) -> (bs, H, N)"]
    KEYS --> DOT1
    KEYS --> DOT2

    DOT1 --> TOP1["topk over N<br>idx1,score1: (bs, H, KNN)"]
    DOT2 --> TOP2["topk over N<br>idx2,score2: (bs, H, KNN)"]

    TOP1 --> COMB["product-key combine<br>pair_scores: (bs, H, KNN, KNN)<br>pair_indices: (bs, H, KNN, KNN)<br>flatten -> (bs, H, KNN^2)"]
    TOP2 --> COMB

    COMB --> TOPF["final topk over KNN^2<br>scores_logits: (bs, H, KNN)<br>indices: (bs, H, KNN)"]

    TOPF --> FLAT2["flatten (bs, H, KNN) -> (bs*H, KNN)<br>scores_logits: (bs*H, KNN)<br>indices: (bs*H, KNN)"]
    FLAT2 --> SOFT["softmax over KNN (per token-head)<br>scores = softmax(scores_logits)<br>scores: (bs*H, KNN)"]

    SOFT --> MERGE["view for value lookup<br>indices: (bs*H, KNN) -> (bs, H*KNN)<br>scores: (bs*H, KNN) -> (bs, H*KNN)"]

    MERGE --> READ["values weighted sum (EmbeddingBag style)<br>values table: (N^2, v_dim)<br>out = sum_j scores[j] * values[indices[j]]<br>out: (bs, v_dim)"]

    READ --> PROJ["project to D<br>out_d: (bs, D)"]
    PROJ --> VDROP["dropout(value)<br>shape: (bs, D)"]
    VDROP --> GATE["optional gating<br>sigmoid(gating_linear(input)) * value<br>shape: (bs, D)"]

    GATE --> UNFLAT["reshape/view back<br>(bs, D) -> (B, T, D)"]
    UNFLAT --> PKM_OUT["pkm_out<br>shape: (B, T, D)"]
  end

  PKM_OUT --> OUT