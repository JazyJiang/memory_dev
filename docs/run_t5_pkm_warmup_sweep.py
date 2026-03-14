#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: PyYAML. Install with `pip install pyyaml`.") from e


_LAYER_CSV_RE = re.compile(r"^\s*\d+\s*(\s*,\s*\d+\s*)*$")


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


def normalize_list(v: Any) -> List[Any]:
    if isinstance(v, list):
        return v
    return [v]


def sanitize_layers(s: str) -> str:
    s = str(s)
    if s.strip() == "":
        return "none"
    return s.replace(",", "--")


def parse_layer_csv(s: str) -> List[int]:
    s = str(s).strip()
    if s == "":
        return []
    if not _LAYER_CSV_RE.match(s):
        raise ValueError(f"Invalid layer csv: {s!r}")
    return [int(x.strip()) for x in s.split(",")]


def infer_runner_defaults(sweep: Dict[str, Any]) -> Dict[str, Any]:
    strategy = str(sweep.get("strategy", "t5_seq2seq"))
    train = sweep.get("train") if isinstance(sweep.get("train"), dict) else {}
    pkm = sweep.get("pkm") if isinstance(sweep.get("pkm"), dict) else {}
    pkm_t5 = pkm.get("t5_seq2seq") if isinstance(pkm.get("t5_seq2seq"), dict) else {}

    lr = train.get("learning_rate", 3.0e-4)
    bs = train.get("batch_size", 256)
    epochs = train.get("epochs", None)

    enc_layers = pkm_t5.get("pk_encoder_layers", "")
    dec_layers = pkm_t5.get("pk_decoder_layers", "2")
    n_keys = pkm_t5.get("pk_mem_n_keys", 128)
    knn = pkm_t5.get("pk_mem_knn", 16)
    topk = pkm_t5.get("pk_topk", 8)

    sweep_params = sweep.get("sweep_params") if isinstance(sweep.get("sweep_params"), dict) else {}
    d0_warmups = normalize_list(sweep_params.get("d0_warmup_epochs", [0, 10]))
    ft_warmups = normalize_list(sweep_params.get("finetune_warmup_epochs", [10]))

    out = {
        "strategy": strategy,
        "train.learning_rate": lr,
        "train.batch_size": bs,
    }

    if epochs is not None:
        out["train.epochs"] = int(epochs)

    out.update({
        "pkm.t5_seq2seq.pk_is_enabled": True,
        "pkm.t5_seq2seq.pk_encoder_layers": str(enc_layers),
        "pkm.t5_seq2seq.pk_decoder_layers": str(dec_layers),
        "pkm.t5_seq2seq.pk_mem_n_keys": int(n_keys),
        "pkm.t5_seq2seq.pk_mem_knn": int(knn),
        "pkm.t5_seq2seq.pk_topk": int(topk),
        "sweep.d0_warmup_epochs": [int(x) for x in d0_warmups],
        "sweep.finetune_warmup_epochs": [int(x) for x in ft_warmups],
    })

    return out


def iter_warmup_grid(base: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    d0s = base["sweep.d0_warmup_epochs"]
    fts = base["sweep.finetune_warmup_epochs"]
    for d0 in d0s:
        for ft in fts:
            out = dict(base)
            out["sweep.d0_warmup_epochs"] = int(d0)
            out["sweep.finetune_warmup_epochs"] = int(ft)
            yield out


def key_for_resume(params: Dict[str, Any]) -> str:
    payload = {
        "train.learning_rate": params.get("train.learning_rate"),
        "train.batch_size": params.get("train.batch_size"),
        "train.epochs": params.get("train.epochs"),
        "pkm.t5_seq2seq.pk_encoder_layers": params.get("pkm.t5_seq2seq.pk_encoder_layers"),
        "pkm.t5_seq2seq.pk_decoder_layers": params.get("pkm.t5_seq2seq.pk_decoder_layers"),
        "pkm.t5_seq2seq.pk_mem_n_keys": params.get("pkm.t5_seq2seq.pk_mem_n_keys"),
        "pkm.t5_seq2seq.pk_topk": params.get("pkm.t5_seq2seq.pk_topk"),
        "sweep.d0_warmup_epochs": params.get("sweep.d0_warmup_epochs"),
        "sweep.finetune_warmup_epochs": params.get("sweep.finetune_warmup_epochs"),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def worker_for_sig(sig: str, num_workers: int) -> int:
    h = hashlib.md5(sig.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % num_workers


def load_seen_sigs(result_jsonl: Optional[str]) -> set[str]:
    if not result_jsonl:
        return set()
    p = Path(result_jsonl)
    if not p.is_file():
        return set()
    seen: set[str] = set()
    for line in p.read_text(errors="ignore").splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        params = obj.get("params")
        if isinstance(params, dict):
            seen.add(key_for_resume(params))
    return seen


def run_cmd(cmd: List[str], env: Dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        p = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            f.write(line)
            f.flush()
            print(line, end="", flush=True)
        p.wait()
    return int(p.returncode or 0)


def export_tb_images(
    tb_log_dir: Path,
    out_dir: Path,
    tag_prefix: str = "PKM/",
    last_k: int = 1,
    stride: int = 1,
) -> List[str]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore
    except Exception:
        return []

    if not tb_log_dir.exists():
        return []

    if stride <= 0:
        stride = 1

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ea = EventAccumulator(
            str(tb_log_dir),
            size_guidance={
                "images": 0,
                "scalars": 0,
                "tensors": 0,
                "histograms": 0,
                "compressedHistograms": 0,
                "audio": 0,
            },
        )
        ea.Reload()
        tags = ea.Tags().get("images", [])
    except Exception:
        return []

    saved: List[str] = []
    for tag in tags:
        if tag_prefix and not str(tag).startswith(tag_prefix):
            continue
        try:
            imgs = ea.Images(tag)
        except Exception:
            continue
        if not imgs:
            continue

        if last_k > 0:
            imgs_sel = imgs[-last_k:]
        else:
            imgs_sel = imgs

        imgs_sel = imgs_sel[::stride]

        safe_tag = str(tag).replace("/", "__")
        is_epoch_tag = str(tag).endswith("_Epoch")
        for j, img in enumerate(imgs_sel):
            step = int(getattr(img, "step", 0) or 0)
            prefix = "epoch" if is_epoch_tag else "step"
            out_path = out_dir / f"{safe_tag}.{prefix}{step}.png"
            if out_path.exists():
                out_path = out_dir / f"{safe_tag}.{prefix}{step}.i{j}.png"
            try:
                out_path.write_bytes(img.encoded_image_string)
            except Exception:
                continue
            saved.append(str(out_path))

    return saved


def main() -> None:
    ap = argparse.ArgumentParser()

    repo_root = Path(__file__).resolve().parents[1]

    ap.add_argument("--sweep_yaml", default=str(repo_root / "docs" / "pkm_warmup_sweep.yaml"))
    ap.add_argument("--config_file", default=str(repo_root / "configs" / "train_t5_pkm_warmup.yaml"))

    ap.add_argument("--dataset", default=os.environ.get("DATASET", "Toys_and_Games"))
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT", str(repo_root / "data")))
    ap.add_argument("--amazon_root", default=os.environ.get("AMAZON_ROOT", str(repo_root / "data")))
    ap.add_argument("--base_model", default=os.environ.get("BASE_MODEL", "google-t5/t5-small"))
    ap.add_argument("--time_range", default=os.environ.get("TIME_RANGE", "2016-10-2018-11"))
    ap.add_argument("--index_file", default=os.environ.get("INDEX_FILE", ".TIGER-index.json"))

    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--wd", type=float, default=float(os.environ.get("WD", "0.001")))
    ap.add_argument("--model_max_length", type=int, default=int(os.environ.get("MODEL_MAX_LENGTH", "512")))
    ap.add_argument("--max_new_tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "10")))
    ap.add_argument("--num_beams", type=int, default=int(os.environ.get("NUM_BEAMS", "20")))
    ap.add_argument("--test_batch_size", type=int, default=int(os.environ.get("TEST_BATCH_SIZE", "8")))

    ap.add_argument("--local_ckpt_root", default=os.environ.get("LOCAL_CKPT_ROOT", f"/tmp/{os.environ.get('USER','user')}/memory_dev_ckpt"))
    ap.add_argument("--cleanup_ckpt", type=int, default=int(os.environ.get("CLEANUP_CKPT", "1")))

    ap.add_argument("--result_jsonl", default=os.environ.get("RESULT_JSONL", None))
    ap.add_argument("--resume_from_result", default=None)

    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--worker_id", type=int, default=0)
    ap.add_argument(
        "--split_mode",
        choices=["round_robin", "hash"],
        default="round_robin",
        help="How to split sweep combos across workers. round_robin guarantees more even splits for small grids.",
    )
    ap.add_argument("--cuda_visible_devices", default=None)
    ap.add_argument("--master_port_base", type=int, default=None)

    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--max_runs", type=int, default=None)
    ap.add_argument("--force_routing_test", action="store_true", help="Apply group mask at test time (forced routing)")

    args = ap.parse_args()

    if args.num_workers <= 0:
        raise ValueError(f"--num_workers must be > 0, got {args.num_workers}")
    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError(f"--worker_id must be in [0, {args.num_workers}), got {args.worker_id}")

    sweep_path = Path(args.sweep_yaml)
    if not sweep_path.is_file():
        raise FileNotFoundError(str(sweep_path))

    cfg_path = Path(args.config_file)
    if not cfg_path.is_file():
        raise FileNotFoundError(str(cfg_path))

    sweep = load_yaml(sweep_path)
    base = infer_runner_defaults(sweep)

    sweep_train = sweep.get("train") if isinstance(sweep.get("train"), dict) else {}
    default_epochs = int(sweep_train.get("epochs", os.environ.get("EPOCHS", "50")))
    epochs = int(args.epochs) if args.epochs is not None else default_epochs

    base.setdefault("train.epochs", epochs)

    if args.result_jsonl is None:
        args.result_jsonl = f"./log/{args.dataset}/sweep_t5_pkm_warmup/result.jsonl"
    result_jsonl = Path(args.result_jsonl)

    status_jsonl = result_jsonl.with_name("sweep_status.jsonl")
    if args.num_workers > 1:
        status_jsonl = status_jsonl.with_name(f"sweep_status.worker{args.worker_id}.jsonl")

    seen = load_seen_sigs(args.resume_from_result or str(result_jsonl))

    master_port_base = args.master_port_base
    if master_port_base is None:
        master_port_base = int(os.environ.get("MASTER_PORT_BASE", "2500"))

    env_base = dict(os.environ)
    env_base["WANDB_MODE"] = "disabled"
    env_base["WANDB_DISABLED"] = "true"
    env_base["TRANSFORMERS_NO_TF"] = env_base.get("TRANSFORMERS_NO_TF", "1")
    env_base["USE_TF"] = env_base.get("USE_TF", "0")
    if args.cuda_visible_devices is not None:
        env_base["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    total = 0
    assigned = 0
    skipped = 0
    launched = 0

    for run_idx, params in enumerate(iter_warmup_grid(base)):
        total += 1
        sig = key_for_resume(params)

        if args.split_mode == "hash":
            assigned_worker = worker_for_sig(sig, args.num_workers)
        else:
            assigned_worker = run_idx % args.num_workers

        if assigned_worker != args.worker_id:
            continue

        assigned += 1

        if sig in seen:
            skipped += 1
            append_jsonl(
                status_jsonl,
                {"created_at": utc_now(), "status": "skipped", "reason": "resume_hit", "params": params, "worker_id": args.worker_id, "num_workers": args.num_workers},
            )
            continue

        if args.max_runs is not None and launched >= args.max_runs:
            break

        d0_warmup = int(params["sweep.d0_warmup_epochs"])
        ft_warmup = int(params["sweep.finetune_warmup_epochs"])

        lr = params["train.learning_rate"]
        bs = params["train.batch_size"]
        enc_tag = sanitize_layers(params.get("pkm.t5_seq2seq.pk_encoder_layers", ""))
        dec_tag = sanitize_layers(params.get("pkm.t5_seq2seq.pk_decoder_layers", ""))
        n_keys = int(params["pkm.t5_seq2seq.pk_mem_n_keys"])
        topk = int(params["pkm.t5_seq2seq.pk_topk"])

        run_epochs = int(params.get("train.epochs", epochs))
        run_tag = f"t5pkm_d0w{d0_warmup}_ftw{ft_warmup}_lr{lr}_bs{bs}_ep{run_epochs}_enc{enc_tag}_dec{dec_tag}_nk{n_keys}_topk{topk}"

        run_ckpt_root = Path(args.local_ckpt_root) / "ckpt" / args.dataset / "sweep_t5_pkm_warmup" / run_tag
        run_log_root = repo_root / "log" / args.dataset / "sweep_t5_pkm_warmup" / run_tag

        train_log_dir = run_log_root / "train"
        test_log_dir = run_log_root / "test"
        train_log_dir.mkdir(parents=True, exist_ok=True)
        test_log_dir.mkdir(parents=True, exist_ok=True)
        result_jsonl.parent.mkdir(parents=True, exist_ok=True)

        params_json = run_log_root / "params.json"
        params_payload = {
            "dataset": args.dataset,
            "strategy": str(params.get("strategy", "t5_seq2seq")),
            "run_tag": run_tag,
            "train.learning_rate": lr,
            "train.batch_size": bs,
            "train.epochs": int(params.get("train.epochs", epochs)),
            "pkm.t5_seq2seq.pk_encoder_layers": params.get("pkm.t5_seq2seq.pk_encoder_layers", ""),
            "pkm.t5_seq2seq.pk_decoder_layers": params.get("pkm.t5_seq2seq.pk_decoder_layers", ""),
            "pkm.t5_seq2seq.pk_mem_n_keys": n_keys,
            "pkm.t5_seq2seq.pk_topk": topk,
            "sweep.d0_warmup_epochs": d0_warmup,
            "sweep.finetune_warmup_epochs": ft_warmup,
            "BASE_MODEL": args.base_model,
            "TIME_RANGE": args.time_range,
            "INDEX_FILE": args.index_file,
            "DATA_ROOT": args.data_root,
            "AMAZON_ROOT": args.amazon_root,
        }
        params_json.write_text(json.dumps(params_payload, ensure_ascii=False, indent=2))

        append_jsonl(
            status_jsonl,
            {"created_at": utc_now(), "status": "start", "run_tag": run_tag, "params": params_payload, "worker_id": args.worker_id, "num_workers": args.num_workers},
        )

        if args.dry_run:
            launched += 1
            append_jsonl(status_jsonl, {"created_at": utc_now(), "status": "dry_run_done", "run_tag": run_tag})
            continue

        t5_overrides_base = [
            "pkm.t5_seq2seq.pk_is_enabled=true",
            f"pkm.t5_seq2seq.pk_encoder_layers={params.get('pkm.t5_seq2seq.pk_encoder_layers','')}",
            f"pkm.t5_seq2seq.pk_decoder_layers={params.get('pkm.t5_seq2seq.pk_decoder_layers','')}",
            f"pkm.t5_seq2seq.pk_mem_n_keys={n_keys}",
            f"pkm.t5_seq2seq.pk_mem_knn={int(params.get('pkm.t5_seq2seq.pk_mem_knn', 16))}",
            f"pkm.t5_seq2seq.pk_topk={topk}",
        ]

        failed = False

        for train_d in (0, 1, 2, 3):
            test_d = train_d + 1
            current_warmup = d0_warmup if train_d == 0 else ft_warmup
            t5_overrides = list(t5_overrides_base) + [f"pkm.t5_seq2seq.pk_warmup_epochs={current_warmup}"]

            cur_ckpt = run_ckpt_root / f"D{train_d}"
            cur_ckpt.mkdir(parents=True, exist_ok=True)

            train_log = train_log_dir / f"{run_tag}_trainD{train_d}.log"
            port = master_port_base + args.worker_id * 200 + run_idx * 10 + train_d
            cmd_train = [
                "torchrun",
                "--nproc_per_node=1",
                f"--master_port={port}",
                "train.py",
                f"config={str(cfg_path)}",
                "strategy=t5_seq2seq",
                f"model.t5_seq2seq.base_model={args.base_model}",
                f"train.output_dir={str(cur_ckpt)}",
                f"train.logging_dir={str(run_log_root / f'D{train_d}' / 'runs')}",
                f"dataset.data_path={args.amazon_root}",
                f"dataset.name={args.dataset}",
                f"dataset.train_file={os.path.join(args.data_root, f'D{train_d}', f'{args.dataset}_5_{args.time_range}.csv')}",
                f"dataset.valid_file={os.path.join(args.data_root, f'D{train_d}', f'{args.dataset}_5_{args.time_range}.csv')}",
                f"dataset.test_file={os.path.join(args.data_root, f'D{train_d}', f'{args.dataset}_5_{args.time_range}.csv')}",
                f"dataset.index_file={args.index_file}",
                f"train.batch_size={bs}",
                f"train.learning_rate={lr}",
                f"train.epochs={int(params.get('train.epochs', epochs))}",
                f"train.weight_decay={args.wd}",
                "train.logging_step=1",
                "train.save_and_eval_strategy=epoch",
                f"train.model_max_length={args.model_max_length}",
            ] + t5_overrides

            rc = run_cmd(cmd_train, env=env_base, log_path=train_log)
            if rc != 0:
                failed = True
                append_jsonl(status_jsonl, {"created_at": utc_now(), "status": "failed", "run_tag": run_tag, "phase": f"train_D{train_d}", "returncode": rc})
                break

            train_tb_dir = run_log_root / f"D{train_d}" / "runs"
            train_png_dir = train_tb_dir / "png"
            train_saved = export_tb_images(train_tb_dir, train_png_dir, tag_prefix="PKM/", last_k=0, stride=1)
            if train_saved:
                append_jsonl(
                    status_jsonl,
                    {
                        "created_at": utc_now(),
                        "status": "train_tb_png_saved",
                        "run_tag": run_tag,
                        "phase": f"train_D{train_d}",
                        "tb_dir": str(train_tb_dir),
                        "png_dir": str(train_png_dir),
                        "png_count": len(train_saved),
                        "png_last": train_saved[-1],
                    },
                )
            else:
                append_jsonl(status_jsonl, {"created_at": utc_now(), "status": "train_tb_png_missing", "run_tag": run_tag, "phase": f"train_D{train_d}", "tb_dir": str(train_tb_dir)})

            group_dir = Path(args.data_root) / f"D{test_d}" / "groups"
            group_files = sorted(glob.glob(str(group_dir / "*.csv")))
            if not group_files:
                append_jsonl(status_jsonl, {"created_at": utc_now(), "status": "warn", "run_tag": run_tag, "phase": f"test_D{train_d}_to_D{test_d}", "warning": f"no_group_files_in={str(group_dir)}"})
                continue

            for group_file in group_files:
                group_name = Path(group_file).stem
                test_log = test_log_dir / f"{run_tag}_trainD{train_d}_testD{test_d}_{group_name}.log"
                cmd_test = [
                    "python",
                    "test.py",
                    f"config={str(cfg_path)}",
                    "model.type=t5_seq2seq",
                    "global.gpu_id=0",
                    f"model.ckpt_path={str(cur_ckpt)}",
                    f"model.tokenizer_path={str(cur_ckpt)}",
                    f"model.base_model={args.base_model}",
                    f"dataset.name={args.dataset}",
                    f"dataset.data_path={args.amazon_root}",
                    f"dataset.train_file={os.path.join(args.data_root, f'D{train_d}', f'{args.dataset}_5_{args.time_range}.csv')}",
                    f"dataset.valid_file={os.path.join(args.data_root, f'D{train_d}', f'{args.dataset}_5_{args.time_range}.csv')}",
                    f"dataset.test_file={group_file}",
                    f"dataset.index_file={args.index_file}",
                    f"test.batch_size={args.test_batch_size}",
                    f"test.num_beams={args.num_beams}",
                    f"test.max_new_tokens={args.max_new_tokens}",
                    "test.filter_items=true",
                    "test.log_pkm_heatmap=false",
                    f"test.pk_force_routing={'true' if args.force_routing_test else 'false'}",
                    f"test.logging_dir={str(run_log_root / f'D{train_d}' / 'test' / 'runs' / f'testD{test_d}_{group_name}')}",
                ] + t5_overrides

                rc2 = run_cmd(cmd_test, env=env_base, log_path=test_log)
                if rc2 != 0:
                    failed = True
                    append_jsonl(status_jsonl, {"created_at": utc_now(), "status": "failed", "run_tag": run_tag, "phase": f"test_{group_name}_D{train_d}_to_D{test_d}", "returncode": rc2})
                    break

            # Only output ONE heatmap for the whole D{test_d}: run an extra test on the ungrouped file.
            if not failed:
                combined_test_file = os.path.join(args.data_root, f"D{test_d}", f"{args.dataset}_5_{args.time_range}.csv")
                combined_run_name = f"testD{test_d}_allgroups"
                combined_test_log = test_log_dir / f"{run_tag}_trainD{train_d}_testD{test_d}_{combined_run_name}.log"
                combined_tb_dir = run_log_root / f"D{train_d}" / "test" / "runs" / combined_run_name

                cmd_test_all = [
                    "python",
                    "test.py",
                    f"config={str(cfg_path)}",
                    "model.type=t5_seq2seq",
                    "global.gpu_id=0",
                    f"model.ckpt_path={str(cur_ckpt)}",
                    f"model.tokenizer_path={str(cur_ckpt)}",
                    f"model.base_model={args.base_model}",
                    f"dataset.name={args.dataset}",
                    f"dataset.data_path={args.amazon_root}",
                    f"dataset.train_file={os.path.join(args.data_root, f'D{train_d}', f'{args.dataset}_5_{args.time_range}.csv')}",
                    f"dataset.valid_file={os.path.join(args.data_root, f'D{train_d}', f'{args.dataset}_5_{args.time_range}.csv')}",
                    f"dataset.test_file={combined_test_file}",
                    f"dataset.index_file={args.index_file}",
                    f"test.batch_size={args.test_batch_size}",
                    f"test.num_beams={args.num_beams}",
                    f"test.max_new_tokens={args.max_new_tokens}",
                    "test.filter_items=true",
                    "test.log_pkm_heatmap=true",
                    f"test.pk_force_routing={'true' if args.force_routing_test else 'false'}",
                    f"test.logging_dir={str(combined_tb_dir)}",
                ] + t5_overrides

                rc3 = run_cmd(cmd_test_all, env=env_base, log_path=combined_test_log)
                if rc3 != 0:
                    failed = True
                    append_jsonl(status_jsonl, {"created_at": utc_now(), "status": "failed", "run_tag": run_tag, "phase": f"test_allgroups_D{train_d}_to_D{test_d}", "returncode": rc3})
                else:
                    combined_png_dir = combined_tb_dir / "png"
                    combined_saved = export_tb_images(combined_tb_dir, combined_png_dir, tag_prefix="PKM/", last_k=1, stride=1)
                    if combined_saved:
                        append_jsonl(
                            status_jsonl,
                            {
                                "created_at": utc_now(),
                                "status": "tb_png_saved",
                                "run_tag": run_tag,
                                "phase": f"test_allgroups_D{train_d}_to_D{test_d}",
                                "tb_dir": str(combined_tb_dir),
                                "png_dir": str(combined_png_dir),
                                "png_count": len(combined_saved),
                                "png_last": combined_saved[-1],
                            },
                        )
                    else:
                        append_jsonl(status_jsonl, {"created_at": utc_now(), "status": "tb_png_missing", "run_tag": run_tag, "phase": f"test_allgroups_D{train_d}_to_D{test_d}", "tb_dir": str(combined_tb_dir)})

            if failed:
                break

        if not failed:
            out = subprocess.run(
                [
                    "python",
                    str(repo_root / "docs" / "write_result_jsonl.py"),
                    "--params_json",
                    str(params_json),
                    "--run_tag",
                    run_tag,
                    "--test_log_glob",
                    str(test_log_dir / f"{run_tag}_trainD*_testD*_*.log"),
                ],
                env=env_base,
                check=False,
                capture_output=True,
                text=True,
            )
            if int(out.returncode) == 0:
                append_jsonl(result_jsonl, json.loads(out.stdout) if out.stdout.strip().startswith("{") else {"run_tag": run_tag, "raw": out.stdout})
                append_jsonl(status_jsonl, {"created_at": utc_now(), "status": "done", "run_tag": run_tag})
            else:
                append_jsonl(status_jsonl, {"created_at": utc_now(), "status": "warn", "run_tag": run_tag, "phase": "write_result_jsonl", "returncode": int(out.returncode), "stderr": out.stderr[-2000:]})

        if args.cleanup_ckpt == 1:
            try:
                subprocess.run(["rm", "-rf", str(run_ckpt_root)], check=False)
            except Exception:
                pass

        launched += 1

    summary = {
        "created_at": utc_now(),
        "status": "sweep_summary",
        "sweep_yaml": str(sweep_path),
        "config_file": str(cfg_path),
        "result_jsonl": str(result_jsonl),
        "status_jsonl": str(status_jsonl),
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "total_combos": total,
        "assigned_to_worker": assigned,
        "skipped_by_resume": skipped,
        "launched_by_worker": launched,
        "dry_run": bool(args.dry_run),
        "max_runs": args.max_runs,
    }
    append_jsonl(status_jsonl, summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()