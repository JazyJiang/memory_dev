import argparse
import itertools
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError(
        "Missing dependency: PyYAML. Install with `pip install pyyaml`."
    ) from e


T5_N_LAYERS = 6  # encoder/decoder layer count for t5-small (0..5)


_LAYER_CSV_RE = re.compile(r"^\s*\d+\s*(\s*,\s*\d+\s*)*$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> Dict[str, Any]:
    obj = yaml.safe_load(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping/dict: {path}")
    return obj


def is_sweep_key(k: str) -> bool:
    return isinstance(k, str) and (k.startswith("train.") or k.startswith("pkm."))


def normalize_value(v: Any) -> List[Any]:
    if isinstance(v, list):
        return v
    return [v]


def parse_layer_csv(s: str) -> List[int]:
    s = str(s).strip()
    if s == "":
        return []
    if not _LAYER_CSV_RE.match(s):
        raise ValueError(f"Invalid layer csv: {s!r}")
    xs = [int(x.strip()) for x in s.split(",")]
    return xs


def validate_combo(params: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []

    # Only support t5_seq2seq sweep in this runner
    if params.get("strategy") not in (None, "t5_seq2seq"):
        reasons.append(f"Unsupported strategy={params.get('strategy')!r}; expected 't5_seq2seq'")

    # Always-on
    if params.get("pkm.t5_seq2seq.pk_is_enabled") not in (True, "true", "True", 1, "1"):
        reasons.append("pkm.t5_seq2seq.pk_is_enabled must be true in this sweep")

    # topk vs n_keys constraint
    try:
        n_keys = int(params["pkm.t5_seq2seq.pk_mem_n_keys"])
        topk = int(params["pkm.t5_seq2seq.pk_topk"])
        if n_keys <= 0:
            reasons.append(f"pk_mem_n_keys must be > 0 (got {n_keys})")
        if topk <= 0:
            reasons.append(f"pk_topk must be > 0 (got {topk})")
        if topk > n_keys:
            reasons.append(f"pk_topk({topk}) > pk_mem_n_keys({n_keys})")
    except Exception as e:
        reasons.append(f"Cannot parse pk_mem_n_keys/pk_topk as int: {e}")

    # mem_k_dim must be even (per your HashingMemory constraints)
    try:
        k_dim = int(params["pkm.t5_seq2seq.pk_mem_k_dim"])
        if k_dim % 2 != 0:
            reasons.append(f"pk_mem_k_dim must be even (got {k_dim})")
        if k_dim <= 0:
            reasons.append(f"pk_mem_k_dim must be > 0 (got {k_dim})")
    except Exception as e:
        reasons.append(f"Cannot parse pk_mem_k_dim as int: {e}")

    # mem_v_dim: -1 or positive
    try:
        v_dim = int(params["pkm.t5_seq2seq.pk_mem_v_dim"])
        if v_dim != -1 and v_dim <= 0:
            reasons.append(f"pk_mem_v_dim must be -1 or > 0 (got {v_dim})")
    except Exception as e:
        reasons.append(f"Cannot parse pk_mem_v_dim as int: {e}")

    # layer indices range check for t5-small: 0..5
    for key in ("pkm.t5_seq2seq.pk_encoder_layers", "pkm.t5_seq2seq.pk_decoder_layers"):
        try:
            layers = parse_layer_csv(params.get(key, ""))
            if any((x < 0 or x >= T5_N_LAYERS) for x in layers):
                reasons.append(f"{key} has out-of-range indices (allowed 0..{T5_N_LAYERS-1}): {layers}")
        except Exception as e:
            reasons.append(f"Invalid {key}: {e}")

    # learning rates / batch size sanity
    try:
        lr = float(params["train.learning_rate"])
        if lr <= 0:
            reasons.append(f"train.learning_rate must be > 0 (got {lr})")
    except Exception as e:
        reasons.append(f"Cannot parse train.learning_rate as float: {e}")

    try:
        bs = int(params["train.batch_size"])
        if bs <= 0:
            reasons.append(f"train.batch_size must be > 0 (got {bs})")
    except Exception as e:
        reasons.append(f"Cannot parse train.batch_size as int: {e}")

    # pk_value_fixed_lr can be "tied" or positive float
    pklr = params.get("pkm.t5_seq2seq.pk_value_fixed_lr")
    if isinstance(pklr, str) and pklr.strip() == "tied":
        pass
    else:
        try:
            pklr_f = float(pklr)
            if pklr_f <= 0:
                reasons.append(f"pk_value_fixed_lr must be 'tied' or > 0 (got {pklr_f})")
        except Exception as e:
            reasons.append(f"Cannot parse pk_value_fixed_lr (expected 'tied' or float): {e}")

    # pk_value_weight_decay numeric and >=0
    try:
        wd = float(params.get("pkm.t5_seq2seq.pk_value_weight_decay", 0.0))
        if wd < 0:
            reasons.append(f"pk_value_weight_decay must be >= 0 (got {wd})")
    except Exception as e:
        reasons.append(f"Cannot parse pk_value_weight_decay as float: {e}")

    # gated/share_values must be bool-ish
    for key in ("pkm.t5_seq2seq.pk_mem_gated", "pkm.t5_seq2seq.pk_mem_share_values"):
        v = params.get(key)
        if v not in (True, False, "true", "false", "True", "False", 1, 0, "1", "0"):
            reasons.append(f"{key} must be boolean-ish (got {v!r})")

    return reasons


def format_env(params: Dict[str, Any]) -> Dict[str, str]:
    def as_env_bool(v: Any) -> str:
        if v in (True, "true", "True", 1, "1"):
            return "1"
        return "0"

    env = dict(os.environ)

    # training core
    env["LR"] = str(params["train.learning_rate"])
    env["BATCH_SIZE"] = str(params["train.batch_size"])

    # placement
    env["T5_PK_ENCODER_LAYERS"] = str(params.get("pkm.t5_seq2seq.pk_encoder_layers", ""))
    env["T5_PK_DECODER_LAYERS"] = str(params.get("pkm.t5_seq2seq.pk_decoder_layers", ""))

    # capacity + retrieval
    env["PK_MEM_N_KEYS"] = str(params["pkm.t5_seq2seq.pk_mem_n_keys"])
    env["PK_TOPK"] = str(params["pkm.t5_seq2seq.pk_topk"])

    # dims
    env["PK_MEM_K_DIM"] = str(params["pkm.t5_seq2seq.pk_mem_k_dim"])
    env["PK_MEM_V_DIM"] = str(params["pkm.t5_seq2seq.pk_mem_v_dim"])

    # optimizer knobs
    env["T5_PK_VALUE_FIXED_LR"] = str(params["pkm.t5_seq2seq.pk_value_fixed_lr"])
    env["T5_PK_VALUE_WEIGHT_DECAY"] = str(params["pkm.t5_seq2seq.pk_value_weight_decay"])

    # switches
    env["PK_MEM_GATED"] = as_env_bool(params["pkm.t5_seq2seq.pk_mem_gated"])
    env["T5_PK_MEM_SHARE_VALUES"] = as_env_bool(params["pkm.t5_seq2seq.pk_mem_share_values"])

    # always clean ckpt for disk safety
    env["CLEANUP_CKPT"] = "1"

    # pass-through optional dataset stuff if provided in YAML
    for k in ("DATASET", "DATA_ROOT", "AMAZON_ROOT", "BASE_MODEL", "TIME_RANGE", "INDEX_FILE"):
        if k in params and params[k] is not None:
            env[k] = str(params[k])

    # result path can be overridden by user via CLI, but leave here as env for bash
    if "RESULT_JSONL" in params and params["RESULT_JSONL"] is not None:
        env["RESULT_JSONL"] = str(params["RESULT_JSONL"])

    return env


def key_for_resume(params: Dict[str, Any], sweep_keys: List[str]) -> str:
    keys = sorted(k for k in params.keys() if is_sweep_key(k))
    payload = {k: params.get(k) for k in sweep_keys}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def iter_grid(sweep: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    ENC_KEY = "pkm.t5_seq2seq.pk_encoder_layers"
    DEC_KEY = "pkm.t5_seq2seq.pk_decoder_layers"
    PLACEMENT_PAIRS_KEY = "placement_pairs"

    N_KEYS_KEY = "pkm.t5_seq2seq.pk_mem_n_keys"
    TOPK_KEY = "pkm.t5_seq2seq.pk_topk"
    CAPACITY_PAIRS_KEY = "capacity_pairs"

    placement_pairs = sweep.get(PLACEMENT_PAIRS_KEY, None)
    capacity_pairs = sweep.get(CAPACITY_PAIRS_KEY, None)

    if placement_pairs is not None:
        for k in (ENC_KEY, DEC_KEY):
            if k in sweep and is_sweep_key(k):
                raise ValueError(
                    f"When `{PLACEMENT_PAIRS_KEY}` is set, do not sweep `{k}` as a list. "
                    f"Remove `{k}` from YAML (or set it to a scalar), and only use `{PLACEMENT_PAIRS_KEY}` "
                    f"to define linked (enc, dec) placements."
                )

    if capacity_pairs is not None:
        for k in (N_KEYS_KEY, TOPK_KEY):
            if k in sweep and is_sweep_key(k):
                raise ValueError(
                    f"When `{CAPACITY_PAIRS_KEY}` is set, do not sweep `{k}` as a list. "
                    f"Remove `{k}` from YAML (or set it to a scalar), and only use `{CAPACITY_PAIRS_KEY}` "
                    f"to define linked (n_keys, topk) capacity pairs."
                )

    fixed: Dict[str, Any] = {}
    for k, v in sweep.items():
        if k in (PLACEMENT_PAIRS_KEY, CAPACITY_PAIRS_KEY):
            continue
        if not is_sweep_key(k):
            fixed[k] = v

    keys: List[str] = []
    values: List[List[Any]] = []
    for sweep_key, sweep_values in sweep.items():
        if sweep_key in (PLACEMENT_PAIRS_KEY, CAPACITY_PAIRS_KEY):
            continue
        if is_sweep_key(sweep_key):
            if placement_pairs is not None and sweep_key in (ENC_KEY, DEC_KEY):
                continue
            if capacity_pairs is not None and sweep_key in (N_KEYS_KEY, TOPK_KEY):
                continue
            keys.append(sweep_key)
            values.append(normalize_value(sweep_values))

    base_product = itertools.product(*values) if values else [()]

    def parse_pair(item: Any) -> Tuple[str, str]:
        if not isinstance(item, dict):
            raise ValueError(f"{PLACEMENT_PAIRS_KEY} item must be a dict, got {type(item)}")
        enc = str(item.get("enc", "")).strip()
        dec = str(item.get("dec", "")).strip()
        return enc, dec

    def parse_capacity_pair(item: Any) -> Tuple[int, int]:
        if not isinstance(item, dict):
            raise ValueError(f"{CAPACITY_PAIRS_KEY} item must be a dict, got {type(item)}")

        n_keys_raw = item.get("n_keys", item.get("pk_mem_n_keys", item.get("pk_mem_n_keys")))
        topk_raw = item.get("topk", item.get("pk_topk", item.get("pk_topk")))

        n_keys = int(n_keys_raw)
        topk = int(topk_raw)
        return n_keys, topk

    parsed_placements: List[Tuple[str, str]] = []
    if placement_pairs is not None:
        parsed_placements = [parse_pair(item) for item in normalize_value(placement_pairs)]
        if not parsed_placements:
            raise ValueError(f"`{PLACEMENT_PAIRS_KEY}` is set but empty.")
        if len(set(parsed_placements)) != len(parsed_placements):
            raise ValueError(f"`{PLACEMENT_PAIRS_KEY}` contains duplicate (enc, dec) pairs: {parsed_placements}")

    parsed_capacities: List[Tuple[int, int]] = []
    if capacity_pairs is not None:
        parsed_capacities = [parse_capacity_pair(item) for item in normalize_value(capacity_pairs)]
        if not parsed_capacities:
            raise ValueError(f"`{CAPACITY_PAIRS_KEY}` is set but empty.")
        if len(set(parsed_capacities)) != len(parsed_capacities):
            raise ValueError(f"`{CAPACITY_PAIRS_KEY}` contains duplicate (n_keys, topk) pairs: {parsed_capacities}")

    for combo in base_product:
        out0 = dict(fixed)
        out0.update({k: combo[i] for i, k in enumerate(keys)})

        candidates: List[Dict[str, Any]] = [out0]

        if placement_pairs is not None:
            next_candidates: List[Dict[str, Any]] = []
            for base in candidates:
                for enc, dec in parsed_placements:
                    out = dict(base)
                    out[ENC_KEY] = enc
                    out[DEC_KEY] = dec
                    next_candidates.append(out)
            candidates = next_candidates

        if capacity_pairs is not None:
            next_candidates = []
            for base in candidates:
                for n_keys, topk in parsed_capacities:
                    out = dict(base)
                    out[N_KEYS_KEY] = n_keys
                    out[TOPK_KEY] = topk
                    next_candidates.append(out)
            candidates = next_candidates

        for out in candidates:
            return_keys = sorted(out.keys())
            yield {k: out[k] for k in return_keys}


def main() -> None:
    ap = argparse.ArgumentParser()

    repo_root = Path(__file__).resolve().parents[1]

    ap.add_argument(
        "--sweep_yaml",
        default=str(repo_root / "docs" / "pkm_t5_seq2seq_hparam_sweep.yaml"),
    )
    ap.add_argument(
        "--runner_sh",
        default=str(repo_root / "docs" / "train_t5_pkm_sweep_quick.sh"),
    )
    ap.add_argument(
        "--result_jsonl",
        default=None,
        help="Override result jsonl path (otherwise use bash default or YAML RESULT_JSONL)",
    )
    ap.add_argument(
        "--resume_from_result",
        default=None,
        help="If set, skip combos already recorded in this jsonl (by params signature).",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print summary and validation stats; do not run.",
    )
    ap.add_argument(
        "--max_runs",
        type=int,
        default=None,
        help="Optional cap for safety; runs first N valid combos.",
    )

    ap.add_argument("--num_workers", type=int, default=1, help="Split sweep across N workers.")
    ap.add_argument("--worker_id", type=int, default=0, help="This worker index in [0, N).")
    ap.add_argument(
        "--cuda_visible_devices",
        default=None,
        help="If set, export CUDA_VISIBLE_DEVICES for the runner bash.",
    )
    ap.add_argument(
        "--master_port_base",
        type=int,
        default=None,
        help="If set, export MASTER_PORT_BASE for the runner bash.",
    )
    ap.add_argument(
        "--shared_result_jsonl",
        action="store_true",
        help="If set, do not suffix RESULT_JSONL/sweep_status.jsonl by worker_id when num_workers>1.",
    )

    args = ap.parse_args()

    if args.num_workers <= 0:
        raise ValueError(f"--num_workers must be > 0, got {args.num_workers}")
    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError(f"--worker_id must be in [0, {args.num_workers}), got {args.worker_id}")

    sweep_path = Path(args.sweep_yaml)
    runner_path = Path(args.runner_sh)
    if not sweep_path.is_file():
        raise FileNotFoundError(str(sweep_path))
    if not runner_path.is_file():
        raise FileNotFoundError(str(runner_path))

    sweep = load_yaml(sweep_path)

    def _suffix_jsonl(path: str, worker_id: int) -> str:
        p = Path(path)
        return str(p.with_name(f"{p.stem}.worker{worker_id}{p.suffix}"))

    if args.result_jsonl is not None:
        sweep["RESULT_JSONL"] = args.result_jsonl

    if args.num_workers > 1 and not args.shared_result_jsonl:
        if "RESULT_JSONL" in sweep and sweep["RESULT_JSONL"] is not None:
            sweep["RESULT_JSONL"] = _suffix_jsonl(str(sweep["RESULT_JSONL"]), args.worker_id)

    sweep_keys = sorted(k for k in sweep.keys() if is_sweep_key(k))

    if "placement_pairs" in sweep:
        for k in ("pkm.t5_seq2seq.pk_encoder_layers", "pkm.t5_seq2seq.pk_decoder_layers"):
            if k not in sweep_keys:
                sweep_keys.append(k)
        sweep_keys = sorted(sweep_keys)

    if "capacity_pairs" in sweep:
        for k in ("pkm.t5_seq2seq.pk_mem_n_keys", "pkm.t5_seq2seq.pk_topk"):
            if k not in sweep_keys:
                sweep_keys.append(k)
        sweep_keys = sorted(sweep_keys)

    seen: set[str] = set()
    if args.resume_from_result:
        rp = Path(args.resume_from_result)
        if rp.is_file():
            for line in rp.read_text(errors="ignore").splitlines():
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                params = obj.get("params")
                if isinstance(params, dict):
                    seen.add(key_for_resume(params, sweep_keys))

    base_status_path = Path(
        args.result_jsonl
        or sweep.get("RESULT_JSONL")
        or f"./log/{sweep.get('DATASET','Toys_and_Games')}/sweep_t5_pkm/result.jsonl"
    )
    sweep_status_jsonl = base_status_path.with_name("sweep_status.jsonl")
    if args.num_workers > 1 and not args.shared_result_jsonl:
        sweep_status_jsonl = sweep_status_jsonl.with_name(f"sweep_status.worker{args.worker_id}.jsonl")

    import hashlib

    def _worker_for_sig(sig: str) -> int:
        h = hashlib.md5(sig.encode("utf-8")).hexdigest()
        return int(h[:8], 16) % args.num_workers

    total = 0
    valid = 0
    skipped = 0
    launched = 0

    assigned_total = 0
    assigned_valid = 0
    assigned_skipped = 0
    assigned_launched = 0

    for params in iter_grid(sweep):
        total += 1
        sig = key_for_resume(params, sweep_keys)

        if _worker_for_sig(sig) != args.worker_id:
            continue

        assigned_total += 1

        if args.resume_from_result and sig in seen:
            skipped += 1
            assigned_skipped += 1
            continue

        reasons = validate_combo(params)
        if reasons:
            skipped += 1
            assigned_skipped += 1
            append_jsonl(
                sweep_status_jsonl,
                {
                    "created_at": utc_now(),
                    "status": "skipped",
                    "reasons": reasons,
                    "worker_id": args.worker_id,
                    "num_workers": args.num_workers,
                    "params": {k: params[k] for k in sorted(k for k in params if is_sweep_key(k) or k in ("DATASET","BASE_MODEL","TIME_RANGE"))},
                },
            )
            continue

        valid += 1
        assigned_valid += 1

        if args.max_runs is not None and assigned_launched >= args.max_runs:
            break

        append_jsonl(
            sweep_status_jsonl,
            {
                "created_at": utc_now(),
                "status": "start",
                "worker_id": args.worker_id,
                "num_workers": args.num_workers,
                "params": {k: params[k] for k in sorted(k for k in params if is_sweep_key(k) or k in ("DATASET","BASE_MODEL","TIME_RANGE"))},
            },
        )

        if args.dry_run:
            launched += 1
            assigned_launched += 1
            continue

        env = format_env(params)
        if args.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        if args.master_port_base is not None:
            env["MASTER_PORT_BASE"] = str(args.master_port_base)

        try:
            proc = subprocess.run(
                ["bash", str(runner_path)],
                env=env,
                check=False,
                stdout=None,
                stderr=None,
            )
            rc = int(proc.returncode)
        except Exception as e:
            append_jsonl(
                sweep_status_jsonl,
                {
                    "created_at": utc_now(),
                    "status": "failed_to_launch",
                    "error": str(e),
                    "worker_id": args.worker_id,
                    "num_workers": args.num_workers,
                    "params": {k: params[k] for k in sorted(k for k in params if is_sweep_key(k) or k in ("DATASET","BASE_MODEL","TIME_RANGE"))},
                },
            )
            launched += 1
            assigned_launched += 1
            continue

        if rc == 0:
            append_jsonl(
                sweep_status_jsonl,
                {
                    "created_at": utc_now(),
                    "status": "done",
                    "returncode": rc,
                    "worker_id": args.worker_id,
                    "num_workers": args.num_workers,
                    "params": {k: params[k] for k in sorted(k for k in params if is_sweep_key(k) or k in ("DATASET","BASE_MODEL","TIME_RANGE"))},
                },
            )
        else:
            append_jsonl(
                sweep_status_jsonl,
                {
                    "created_at": utc_now(),
                    "status": "failed",
                    "returncode": rc,
                    "worker_id": args.worker_id,
                    "num_workers": args.num_workers,
                    "params": {k: params[k] for k in sorted(k for k in params if is_sweep_key(k) or k in ("DATASET","BASE_MODEL","TIME_RANGE"))},
                },
            )

        launched += 1
        assigned_launched += 1

    summary = {
        "created_at": utc_now(),
        "status": "sweep_summary",
        "sweep_yaml": str(sweep_path),
        "runner_sh": str(runner_path),
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "total_combos_enumerated": total,
        "combos_assigned_to_this_worker": assigned_total,
        "valid_combos_total": valid,
        "valid_combos_assigned": assigned_valid,
        "skipped_combos_total": skipped,
        "skipped_combos_assigned": assigned_skipped,
        "launched_combos_total": launched,
        "launched_combos_assigned": assigned_launched,
        "dry_run": bool(args.dry_run),
        "max_runs": args.max_runs,
        "resume_from_result": args.resume_from_result,
        "result_jsonl": str(sweep.get("RESULT_JSONL")) if sweep.get("RESULT_JSONL") is not None else None,
        "sweep_status_jsonl": str(sweep_status_jsonl),
    }
    append_jsonl(sweep_status_jsonl, summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()