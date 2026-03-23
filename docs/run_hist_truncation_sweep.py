"""
History-truncation sweep runner (Exp 2).

Sweeps TEST_MAX_HIS_LEN x period x group over a fixed checkpoint,
distributing work across workers by MD5 hash (same pattern as run_t5_pkm_sweep.py).

Usage (single machine):
    python docs/run_hist_truncation_sweep.py

Multi-worker (one process per GPU):
    CUDA_VISIBLE_DEVICES=0 python docs/run_hist_truncation_sweep.py --num_workers 4 --worker_id 0
    CUDA_VISIBLE_DEVICES=1 python docs/run_hist_truncation_sweep.py --num_workers 4 --worker_id 1
    ...

The sweep is configured via a YAML file (default: docs/hist_truncation_sweep.yaml).
"""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: PyYAML. Install with `pip install pyyaml`.") from e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> Dict[str, Any]:
    obj = yaml.safe_load(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping/dict: {path}")
    return obj


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def combo_sig(combo: Dict[str, Any]) -> str:
    """Stable string signature used for worker assignment and resume dedup."""
    return json.dumps(combo, sort_keys=True, ensure_ascii=False)


def worker_for_sig(sig: str, num_workers: int) -> int:
    h = hashlib.md5(sig.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % num_workers


# ---------------------------------------------------------------------------
# Combo enumeration
# ---------------------------------------------------------------------------

def iter_combos(cfg: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield one dict per (ckpt_path, period, group, hist_len) combination.

    If 'checkpoints' is a list of dicts, each entry can override PKM params.
    If 'ckpt_path' is a single string, it is wrapped into a single-entry list.

    Combo keys (all str for JSON stability):
        ckpt_path, period, group, hist_len
        + all PKM/data params needed by test_hist_truncation.sh
    """
    # Resolve checkpoint list
    if "checkpoints" in cfg and cfg["checkpoints"]:
        ckpt_entries: List[Dict[str, Any]] = []
        for entry in cfg["checkpoints"]:
            if isinstance(entry, str):
                ckpt_entries.append({"ckpt_path": entry})
            elif isinstance(entry, dict):
                ckpt_entries.append(entry)
            else:
                raise ValueError(f"Invalid checkpoint entry: {entry!r}")
    elif "ckpt_path" in cfg:
        ckpt_entries = [{"ckpt_path": cfg["ckpt_path"]}]
    else:
        raise ValueError("YAML must set 'ckpt_path' or 'checkpoints'.")

    # Top-level PKM defaults (can be overridden per checkpoint entry)
    pkm_defaults: Dict[str, Any] = cfg.get("pkm", {})

    # Sweep axes
    hist_lens: List[int] = [int(x) for x in cfg.get("test_max_his_len", [1, 2, 3, 4, 5, 7, 10])]
    periods: List[int] = [int(x) for x in cfg.get("periods", [1, 2, 3, 4])]
    groups: List[int] = [int(x) for x in cfg.get("groups", [1, 2, 3, 4, 5])]

    # Data config
    dataset: str = str(cfg.get("dataset", "Toys_and_Games"))
    data_root: str = str(cfg.get("data_root", ""))
    time_range: str = str(cfg.get("time_range", "2016-10-2018-11"))
    base_model: str = str(cfg.get("base_model", "google-t5/t5-small"))
    index_file: str = str(cfg.get("index_file", ".TIGER-index.json"))
    config_file: str = str(cfg.get("config_file", ""))

    # Group-file pattern: data_root/D{period}/groups/test_user_group{group}.csv
    group_file_pattern: str = str(
        cfg.get(
            "group_file_pattern",
            "{data_root}/D{period}/groups/test_user_group{group}.csv",
        )
    )

    for ckpt_entry in ckpt_entries:
        ckpt_path = str(ckpt_entry["ckpt_path"])

        # Merge PKM params: top-level defaults overridden by per-checkpoint values
        pkm = dict(pkm_defaults)
        for k, v in ckpt_entry.items():
            if k != "ckpt_path":
                pkm[k] = v

        for period in periods:
            for group in groups:
                group_file = group_file_pattern.format(
                    data_root=data_root,
                    period=period,
                    group=group,
                )
                for hist_len in hist_lens:
                    combo: Dict[str, Any] = {
                        "ckpt_path": ckpt_path,
                        "period": period,
                        "group": group,
                        "hist_len": hist_len,
                        "group_file": group_file,
                        # data
                        "dataset": dataset,
                        "data_root": data_root,
                        "time_range": time_range,
                        "base_model": base_model,
                        "index_file": index_file,
                        "config_file": config_file,
                        # pkm (flattened for signature stability)
                        "pk_encoder_layers": str(pkm.get("t5_pk_encoder_layers", "")),
                        "pk_decoder_layers": str(pkm.get("t5_pk_decoder_layers", "2")),
                        "pk_mem_n_keys": int(pkm.get("pk_mem_n_keys", 128)),
                        "pk_topk": int(pkm.get("pk_topk", 8)),
                        "pk_mem_heads": int(pkm.get("pk_mem_heads", 4)),
                        "pk_mem_k_dim": int(pkm.get("pk_mem_k_dim", 512)),
                        "pk_mem_v_dim": int(pkm.get("pk_mem_v_dim", -1)),
                        "pk_mem_gated": int(pkm.get("pk_mem_gated", 0)),
                        "pk_mem_share_values": int(pkm.get("pk_mem_share_values", 0)),
                        "pk_value_fixed_lr": float(pkm.get("pk_value_fixed_lr", 0.001)),
                        "pk_is_enabled": str(pkm.get("pk_is_enabled", "true")).lower(),
                    }
                    yield combo


# ---------------------------------------------------------------------------
# Env builder
# ---------------------------------------------------------------------------

def build_env(combo: Dict[str, Any], result_jsonl: str) -> Dict[str, str]:
    env = dict(os.environ)

    env["CKPT_PATH"] = combo["ckpt_path"]
    env["GROUP_FILE"] = combo["group_file"]
    env["TEST_MAX_HIS_LEN"] = str(combo["hist_len"])
    env["RESULT_JSONL"] = result_jsonl

    env["DATASET"] = combo["dataset"]
    env["DATA_ROOT"] = combo["data_root"]
    env["AMAZON_ROOT"] = combo["data_root"]
    env["BASE_MODEL"] = combo["base_model"]
    env["TIME_RANGE"] = combo["time_range"]
    env["INDEX_FILE"] = combo["index_file"]
    if combo.get("config_file"):
        env["CONFIG_FILE"] = combo["config_file"]

    env["T5_PK_ENCODER_LAYERS"] = combo["pk_encoder_layers"]
    env["T5_PK_DECODER_LAYERS"] = combo["pk_decoder_layers"]
    env["PK_MEM_N_KEYS"] = str(combo["pk_mem_n_keys"])
    env["PK_TOPK"] = str(combo["pk_topk"])
    env["PK_MEM_HEADS"] = str(combo["pk_mem_heads"])
    env["PK_MEM_K_DIM"] = str(combo["pk_mem_k_dim"])
    env["PK_MEM_V_DIM"] = str(combo["pk_mem_v_dim"])
    env["PK_MEM_GATED"] = str(combo["pk_mem_gated"])
    env["T5_PK_MEM_SHARE_VALUES"] = str(combo["pk_mem_share_values"])
    env["T5_PK_VALUE_FIXED_LR"] = str(combo["pk_value_fixed_lr"])
    env["T5_PK_IS_ENABLED"] = combo["pk_is_enabled"]

    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--sweep_yaml",
        default=str(repo_root / "docs" / "hist_truncation_sweep.yaml"),
        help="YAML file defining sweep config (default: docs/hist_truncation_sweep.yaml)",
    )
    ap.add_argument(
        "--runner_sh",
        default=str(repo_root / "docs" / "test_hist_truncation.sh"),
        help="Bash script to run for each combo (default: docs/test_hist_truncation.sh)",
    )
    ap.add_argument(
        "--result_jsonl",
        default=None,
        help="Override result JSONL path (otherwise taken from YAML or default).",
    )
    ap.add_argument(
        "--resume_from_result",
        default=None,
        help="If set, skip combos whose signature already appears in this JSONL.",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print combos without running; useful for sanity-checking.",
    )
    ap.add_argument(
        "--max_runs",
        type=int,
        default=None,
        help="Cap this worker to at most N runs (safety valve).",
    )
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--worker_id", type=int, default=0)
    ap.add_argument(
        "--cuda_visible_devices",
        default=None,
        help="If set, override CUDA_VISIBLE_DEVICES for the runner bash.",
    )
    ap.add_argument(
        "--shared_result_jsonl",
        action="store_true",
        help="Do not suffix result / status JSONL paths by worker_id when num_workers > 1.",
    )
    args = ap.parse_args()

    if args.num_workers <= 0:
        raise ValueError(f"--num_workers must be > 0, got {args.num_workers}")
    if not (0 <= args.worker_id < args.num_workers):
        raise ValueError(f"--worker_id must be in [0, {args.num_workers}), got {args.worker_id}")

    sweep_path = Path(args.sweep_yaml)
    runner_path = Path(args.runner_sh)
    if not sweep_path.is_file():
        raise FileNotFoundError(f"sweep YAML not found: {sweep_path}")
    if not runner_path.is_file():
        raise FileNotFoundError(f"runner bash not found: {runner_path}")

    cfg = load_yaml(sweep_path)

    # Resolve result JSONL
    result_jsonl_base: str = (
        args.result_jsonl
        or str(cfg.get("result_jsonl", "./log/hist_truncation/result.jsonl"))
    )

    def _suffix(path_str: str, wid: int) -> str:
        p = Path(path_str)
        return str(p.with_name(f"{p.stem}.worker{wid}{p.suffix}"))

    if args.num_workers > 1 and not args.shared_result_jsonl:
        result_jsonl_str = _suffix(result_jsonl_base, args.worker_id)
    else:
        result_jsonl_str = result_jsonl_base

    status_jsonl = Path(result_jsonl_str).with_name("sweep_status.jsonl")
    if args.num_workers > 1 and not args.shared_result_jsonl:
        status_jsonl = status_jsonl.with_name(f"sweep_status.worker{args.worker_id}.jsonl")

    # Resume: load already-seen signatures
    seen: set = set()
    if args.resume_from_result:
        rp = Path(args.resume_from_result)
        if rp.is_file():
            for line in rp.read_text(errors="ignore").splitlines():
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sig = obj.get("sig")
                if sig:
                    seen.add(sig)

    # Counters
    total = assigned = skipped = launched = 0

    for combo in iter_combos(cfg):
        total += 1
        sig = combo_sig({k: combo[k] for k in ["ckpt_path", "period", "group", "hist_len"]})

        if worker_for_sig(sig, args.num_workers) != args.worker_id:
            continue

        assigned += 1

        if args.resume_from_result and sig in seen:
            skipped += 1
            continue

        if args.max_runs is not None and launched >= args.max_runs:
            break

        run_record: Dict[str, Any] = {
            "sig": sig,
            "ckpt_path": combo["ckpt_path"],
            "period": combo["period"],
            "group": combo["group"],
            "hist_len": combo["hist_len"],
            "worker_id": args.worker_id,
            "num_workers": args.num_workers,
        }

        append_jsonl(status_jsonl, {"created_at": utc_now(), "status": "start", **run_record})

        if args.dry_run:
            print(json.dumps({"dry_run": True, **run_record}, ensure_ascii=False))
            launched += 1
            continue

        env = build_env(combo, result_jsonl_str)
        if args.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

        try:
            proc = subprocess.run(
                ["bash", str(runner_path)],
                env=env,
                check=False,
            )
            rc = int(proc.returncode)
        except Exception as exc:
            append_jsonl(
                status_jsonl,
                {"created_at": utc_now(), "status": "failed_to_launch", "error": str(exc), **run_record},
            )
            launched += 1
            continue

        status = "done" if rc == 0 else "failed"
        append_jsonl(
            status_jsonl,
            {"created_at": utc_now(), "status": status, "returncode": rc, **run_record},
        )
        launched += 1

    summary = {
        "created_at": utc_now(),
        "status": "sweep_summary",
        "sweep_yaml": str(sweep_path),
        "runner_sh": str(runner_path),
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "total_combos": total,
        "assigned_combos": assigned,
        "skipped_combos": skipped,
        "launched_combos": launched,
        "dry_run": bool(args.dry_run),
        "max_runs": args.max_runs,
        "result_jsonl": result_jsonl_str,
        "status_jsonl": str(status_jsonl),
    }
    append_jsonl(status_jsonl, summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
