#!/usr/bin/env python3
"""
Training runner for delta-set analysis.

Sweeps over (train_max_his_len, test_max_his_len, pk_is_enabled) combos.
For each combo:
  1. Train D0→D3, test D1→D4 (group files)
  2. Save per-sample prediction JSONL alongside aggregate logs
  3. Delete checkpoint immediately after testing (ckpt lives in $PWD/ckpt/)

Usage:
    CUDA_VISIBLE_DEVICES=0 python docs/run_delta_set_sweep.py --num_workers 1 --worker_id 0

The sweep is defined inline in _DELTA_COMBOS below (no external YAML needed).
"""

import argparse
import glob
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Sweep definition
# Each entry is one experiment configuration.
# Fields: train_h, test_h, pk (bool), label, routing (dict, optional)
# ---------------------------------------------------------------------------
_DELTA_COMBOS = [
    # === Group 0: Baselines (from previous sweep) ===
    {"train_h": 2,  "test_h": 2,  "pk": False, "label": "h2_t5"},
    {"train_h": 10, "test_h": 10, "pk": False, "label": "h10_t5"},
    {"train_h": 10, "test_h": 10, "pk": True,  "label": "h10_pkm"},

    # === Group 1: Cross-Attention Routing (FFN only) ===
    # 1a: routing + standard FFN, h=10
    {"train_h": 10, "test_h": 10, "pk": False, "label": "h10_route_ffn",
     "routing": {"enabled": True, "early_layers": [3], "recent_history_len": 2}},
    # 1b: routing + h=2 (sanity check — early mask is empty)
    {"train_h": 2,  "test_h": 2,  "pk": False, "label": "h2_route_ffn",
     "routing": {"enabled": True, "early_layers": [3], "recent_history_len": 2}},

    # === Group 2: Routing + PKM ===
    # 2a: routing + PKM on early layer
    {"train_h": 10, "test_h": 10, "pk": True,  "label": "h10_route_pkm",
     "routing": {"enabled": True, "early_layers": [3], "recent_history_len": 2}},
    # 2b: routing + PKM + gated residual + L1
    {"train_h": 10, "test_h": 10, "pk": True,  "label": "h10_route_pkm_gate",
     "routing": {"enabled": True, "early_layers": [3], "recent_history_len": 2,
                 "gate_l1_weight": 0.01},
     "pk_extra": {"pk_mem_gated": True}},

    # === Group 3: Auxiliary Loss ===
    # 3a: routing + FFN + aux loss
    {"train_h": 10, "test_h": 10, "pk": False, "label": "h10_route_ffn_aux",
     "routing": {"enabled": True, "early_layers": [3], "recent_history_len": 2,
                 "aux_loss_weight": 0.1}},
    # 3b: routing + PKM + gate + aux loss (full combo)
    {"train_h": 10, "test_h": 10, "pk": True,  "label": "h10_route_pkm_gate_aux",
     "routing": {"enabled": True, "early_layers": [3], "recent_history_len": 2,
                 "gate_l1_weight": 0.01, "aux_loss_weight": 0.1},
     "pk_extra": {"pk_mem_gated": True}},
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_cmd(cmd: List[str], env: Dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        p = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            f.write(line)
            f.flush()
            print(line, end="", flush=True)
        p.wait()
    return int(p.returncode or 0)


def key_for_combo(combo: Dict[str, Any]) -> str:
    return json.dumps({k: combo[k] for k in sorted(combo)}, sort_keys=True)


def load_seen(result_jsonl: Path) -> set:
    if not result_jsonl.is_file():
        return set()
    seen = set()
    for line in result_jsonl.read_text(errors="ignore").splitlines():
        try:
            obj = json.loads(line)
            seen.add(obj.get("label", ""))
        except Exception:
            pass
    return seen


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.environ.get("DATASET", "Toys_and_Games"))
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT", str(repo_root / "data")))
    ap.add_argument("--amazon_root", default=os.environ.get("AMAZON_ROOT", str(repo_root / "data")))
    ap.add_argument("--base_model", default=os.environ.get("BASE_MODEL", "google-t5/t5-small"))
    ap.add_argument("--time_range", default=os.environ.get("TIME_RANGE", "2016-10-2018-11"))
    ap.add_argument("--index_file", default=os.environ.get("INDEX_FILE", ".TIGER-index.json"))
    ap.add_argument("--config_file", default=str(repo_root / "configs" / "train_t5_pkm_warmup.yaml"))

    ap.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "50")))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("LR", "1e-3")))
    ap.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", "512")))
    ap.add_argument("--wd", type=float, default=float(os.environ.get("WD", "0.001")))
    ap.add_argument("--model_max_length", type=int, default=int(os.environ.get("MODEL_MAX_LENGTH", "512")))
    ap.add_argument("--max_new_tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "10")))
    ap.add_argument("--num_beams", type=int, default=int(os.environ.get("NUM_BEAMS", "20")))
    ap.add_argument("--test_batch_size", type=int, default=int(os.environ.get("TEST_BATCH_SIZE", "8")))

    # PKM fixed hyperparams (used only when pk=True)
    ap.add_argument("--pk_decoder_layers", default=os.environ.get("PK_DECODER_LAYERS", "3"))
    ap.add_argument("--pk_n_keys", type=int, default=int(os.environ.get("PK_N_KEYS", "128")))
    ap.add_argument("--pk_topk", type=int, default=int(os.environ.get("PK_TOPK", "32")))
    ap.add_argument("--pk_k_dim", type=int, default=int(os.environ.get("PK_K_DIM", "512")))

    # ckpt stored in $PWD/ckpt (not /tmp), deleted after use
    ap.add_argument("--ckpt_root", default=os.environ.get("CKPT_ROOT", str(Path.cwd() / "ckpt")))

    ap.add_argument("--result_jsonl", default=None)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--worker_id", type=int, default=0)
    ap.add_argument("--master_port_base", type=int, default=int(os.environ.get("MASTER_PORT_BASE", "2500")))
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Skip combos already in result_jsonl")
    args = ap.parse_args()

    if args.result_jsonl is None:
        args.result_jsonl = f"./log/{args.dataset}/delta_set_sweep/result.jsonl"
    result_jsonl = Path(args.result_jsonl)
    status_jsonl = result_jsonl.with_name("status.jsonl")
    per_sample_root = result_jsonl.parent / "per_sample"

    seen_labels = load_seen(result_jsonl) if args.resume else set()

    env_base = dict(os.environ)
    env_base["WANDB_MODE"] = "disabled"
    env_base["WANDB_DISABLED"] = "true"
    env_base["TRANSFORMERS_NO_TF"] = "1"
    env_base["USE_TF"] = "0"

    combos_for_worker = [
        c for i, c in enumerate(_DELTA_COMBOS)
        if i % args.num_workers == args.worker_id
    ]

    for combo in combos_for_worker:
        label = combo["label"]
        train_h = int(combo["train_h"])
        test_h = int(combo["test_h"])
        pk = bool(combo["pk"])

        if label in seen_labels:
            print(f"[skip] {label} already in result_jsonl")
            continue

        run_tag = f"delta_{label}_ep{args.epochs}_lr{args.lr}_bs{args.batch_size}"
        run_ckpt_root = Path(args.ckpt_root) / args.dataset / "delta_set" / run_tag
        run_log_root = repo_root / "log" / args.dataset / "delta_set_sweep" / run_tag
        train_log_dir = run_log_root / "train"
        test_log_dir = run_log_root / "test"
        train_log_dir.mkdir(parents=True, exist_ok=True)
        test_log_dir.mkdir(parents=True, exist_ok=True)

        # Write params.json (required by write_result_jsonl.py)
        params_json = run_log_root / "params.json"
        run_log_root.mkdir(parents=True, exist_ok=True)
        params_json.write_text(json.dumps({
            "dataset": args.dataset,
            "label": label,
            "run_tag": run_tag,
            "train_max_his_len": train_h,
            "test_max_his_len": test_h,
            "pk_is_enabled": pk,
            "routing": routing,
            "train.learning_rate": args.lr,
            "train.batch_size": args.batch_size,
            "train.epochs": args.epochs,
        }, ensure_ascii=False, indent=2))

        append_jsonl(status_jsonl, {
            "created_at": utc_now(), "status": "start",
            "label": label, "combo": combo, "run_tag": run_tag,
        })

        if args.dry_run:
            print(f"[dry_run] would run: {run_tag}")
            continue

        # Fixed PKM overrides
        pk_overrides = [
            f"pkm.t5_seq2seq.pk_is_enabled={'true' if pk else 'false'}",
            f"pkm.t5_seq2seq.pk_encoder_layers=",
            f"pkm.t5_seq2seq.pk_decoder_layers={args.pk_decoder_layers if pk else ''}",
            f"pkm.t5_seq2seq.pk_mem_n_keys={args.pk_n_keys}",
            f"pkm.t5_seq2seq.pk_topk={args.pk_topk}",
            f"pkm.t5_seq2seq.pk_mem_k_dim={args.pk_k_dim}",
            "pkm.t5_seq2seq.pk_warmup_epochs=0",
        ]

        # Per-combo PKM extras (e.g., pk_mem_gated)
        pk_extra = combo.get("pk_extra", {})
        for k, v in pk_extra.items():
            pk_overrides.append(f"pkm.t5_seq2seq.{k}={str(v).lower() if isinstance(v, bool) else v}")

        # Routing overrides
        routing = combo.get("routing", {})
        routing_overrides = []
        if routing.get("enabled", False):
            routing_overrides = [
                "routing.enabled=true",
                f"routing.early_layers=[{','.join(str(x) for x in routing.get('early_layers', [3]))}]",
                f"routing.recent_history_len={routing.get('recent_history_len', 2)}",
                f"routing.gate_l1_weight={routing.get('gate_l1_weight', 0.0)}",
                f"routing.aux_loss_weight={routing.get('aux_loss_weight', 0.0)}",
            ]
        else:
            routing_overrides = ["routing.enabled=false"]

        failed = False

        for train_d in range(4):  # D0, D1, D2, D3
            test_d = train_d + 1

            cur_ckpt = run_ckpt_root / f"D{train_d}"
            cur_ckpt.mkdir(parents=True, exist_ok=True)

            # ── Train ──
            train_log = train_log_dir / f"{run_tag}_trainD{train_d}.log"
            port = args.master_port_base + args.worker_id * 100 + train_d
            cmd_train = [
                "torchrun", "--nproc_per_node=1", f"--master_port={port}",
                "train.py",
                f"config={args.config_file}",
                "strategy=t5_seq2seq",
                f"model.t5_seq2seq.base_model={args.base_model}",
                f"train.output_dir={cur_ckpt}",
                f"train.logging_dir={run_log_root / f'D{train_d}' / 'runs'}",
                f"dataset.data_path={args.amazon_root}",
                f"dataset.name={args.dataset}",
                f"dataset.train_file={args.data_root}/D{train_d}/{args.dataset}_5_{args.time_range}.csv",
                f"dataset.valid_file={args.data_root}/D{train_d}/{args.dataset}_5_{args.time_range}.csv",
                f"dataset.test_file={args.data_root}/D{train_d}/{args.dataset}_5_{args.time_range}.csv",
                f"dataset.index_file={args.index_file}",
                f"dataset.train_max_his_len={train_h}",
                f"train.batch_size={args.batch_size}",
                f"train.learning_rate={args.lr}",
                f"train.epochs={args.epochs}",
                f"train.weight_decay={args.wd}",
                "train.logging_step=1",
                "train.save_and_eval_strategy=epoch",
                f"train.model_max_length={args.model_max_length}",
            ] + pk_overrides + routing_overrides

            rc = run_cmd(cmd_train, env_base, train_log)
            if rc != 0:
                failed = True
                append_jsonl(status_jsonl, {
                    "created_at": utc_now(), "status": "failed",
                    "label": label, "phase": f"train_D{train_d}", "returncode": rc,
                })
                break

            # ── Test on each group file ──
            group_dir = Path(args.data_root) / f"D{test_d}" / "groups"
            group_files = sorted(glob.glob(str(group_dir / "*.csv")))
            if not group_files:
                append_jsonl(status_jsonl, {
                    "created_at": utc_now(), "status": "warn",
                    "label": label, "warning": f"no group files in {group_dir}",
                })

            for group_file in group_files:
                group_name = Path(group_file).stem
                test_log = test_log_dir / f"{run_tag}_trainD{train_d}_testD{test_d}_{group_name}.log"

                # per-sample JSONL path
                per_sample_jsonl = per_sample_root / label / f"trainD{train_d}_testD{test_d}_{group_name}.jsonl"

                cmd_test = [
                    "python", "test.py",
                    f"config={args.config_file}",
                    "model.type=t5_seq2seq",
                    "global.gpu_id=0",
                    f"model.ckpt_path={cur_ckpt}",
                    f"model.tokenizer_path={cur_ckpt}",
                    f"model.base_model={args.base_model}",
                    f"dataset.name={args.dataset}",
                    f"dataset.data_path={args.amazon_root}",
                    f"dataset.train_file={args.data_root}/D{train_d}/{args.dataset}_5_{args.time_range}.csv",
                    f"dataset.valid_file={args.data_root}/D{train_d}/{args.dataset}_5_{args.time_range}.csv",
                    f"dataset.test_file={group_file}",
                    f"dataset.index_file={args.index_file}",
                    f"dataset.test_max_his_len={test_h}",
                    f"test.batch_size={args.test_batch_size}",
                    f"test.num_beams={args.num_beams}",
                    f"test.max_new_tokens={args.max_new_tokens}",
                    "test.filter_items=true",
                    "test.log_pkm_heatmap=false",
                    f"test.per_sample_jsonl={per_sample_jsonl}",
                ] + pk_overrides + routing_overrides

                rc2 = run_cmd(cmd_test, env_base, test_log)
                if rc2 != 0:
                    failed = True
                    append_jsonl(status_jsonl, {
                        "created_at": utc_now(), "status": "failed",
                        "label": label, "phase": f"test_{group_name}_D{train_d}→D{test_d}", "returncode": rc2,
                    })
                    break

            if failed:
                break

            # ── Delete checkpoint for this period ──
            try:
                subprocess.run(["rm", "-rf", str(cur_ckpt)], check=False)
                print(f"[cleanup] deleted {cur_ckpt}")
            except Exception as e:
                print(f"[cleanup] warning: {e}")

        # If all periods done, collect aggregate metrics and write to result_jsonl
        if not failed:
            out = subprocess.run(
                [
                    "python", str(repo_root / "docs" / "write_result_jsonl.py"),
                    "--params_json", str(params_json),
                    "--run_tag", run_tag,
                    "--test_log_glob", str(test_log_dir / f"{run_tag}_trainD*_testD*_*.log"),
                ],
                env=env_base, capture_output=True, text=True, check=False,
            )

            # Write a summary entry even if write_result_jsonl fails
            summary_entry = {
                "created_at": utc_now(),
                "label": label,
                "combo": combo,
                "run_tag": run_tag,
                "per_sample_root": str(per_sample_root / label),
            }
            if out.returncode == 0 and out.stdout.strip().startswith("{"):
                summary_entry["metrics"] = json.loads(out.stdout)
            append_jsonl(result_jsonl, summary_entry)
            append_jsonl(status_jsonl, {
                "created_at": utc_now(), "status": "done", "label": label,
            })
            print(f"[done] {label}")
        else:
            # Clean up remaining ckpt dirs on failure
            try:
                subprocess.run(["rm", "-rf", str(run_ckpt_root)], check=False)
            except Exception:
                pass

    print("Sweep complete.")


if __name__ == "__main__":
    main()
