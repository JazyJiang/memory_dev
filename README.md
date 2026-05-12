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

## Supported Datasets

| Dataset | Time Range | Approx. Size |
|---------|-----------|--------------|
| `Toys_and_Games` | 2016-10 ~ 2018-11 | ~170K interactions |
| `Video_Games` | 2012-10 ~ 2018-11 | ~230K interactions |
| `CDs_and_Vinyl` | 2014-10 ~ 2018-11 | ~350K interactions |
| `Books` | 2016-10 ~ 2018-11 | ~1.5M interactions |

## Experiment Methods

```bash
bash run.sh <dataset> <method> <gpu_id>
```

| Method | Description |
|--------|-------------|
| `baseline_h2` | T5, history truncated to 2 items |
| `baseline_h10` | T5, history truncated to 10 items |
| `routing` | Cross-attention routing (early vs recent history) |
| `routing_aux` | Routing + auxiliary prediction loss |
| `pkm` | T5 + Product-Key Memory |
| `pkm_routing_aux` | Full: PKM + Routing + Gated + Aux Loss |
| `all` | Run all of the above sequentially |

## CL Protocol

Each experiment runs the full continual learning chain:
```
D0 train → test on D1 (per group)
D1 finetune → test on D2 (per group)
D2 finetune → test on D3 (per group)
D3 finetune → test on D4 (per group)
```

Results are saved to `log/<dataset>/delta_set_sweep/result.jsonl`.

## Multi-GPU Parallel

Run different methods on different GPUs:
```bash
bash run.sh Toys_and_Games baseline_h10 0 &
bash run.sh Toys_and_Games routing_aux 1 &
bash run.sh Toys_and_Games pkm 2 &
wait
```

## Project Structure

```
├── run.sh                    # Unified experiment launcher
├── scripts/
│   └── setup_data.sh         # Data download + processing pipeline
├── train.py                  # Training entry point
├── test.py                   # Evaluation (beam search + ranking)
├── data.py                   # Dataset classes
├── models/
│   ├── routed_t5.py          # Cross-attention routing + aux head
│   └── decoder_only/         # Decoder-only Transformer (optional)
├── pkm/                      # Product-Key Memory module
├── RQ-VAE/                   # Semantic ID generation
├── configs/                  # Training configs (OmegaConf YAML)
├── docs/
│   └── run_delta_set_sweep.py  # Sweep runner (called by run.sh)
└── data/                     # Processed data (after setup_data.sh)
```
