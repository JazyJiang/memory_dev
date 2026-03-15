#!/usr/bin/env python3
"""Full-retrain sweep runner.

Each stage trains from scratch on all data accumulated up to that point:
  D0           -> test D1
  D0+D1        -> test D2
  D0+D1+D2     -> test D3
  D0+D1+D2+D3  -> test D4

Use --with_pkm to enable PKM memory layer.
Use --force_routing_test to apply group mask at test time (PKM only).
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: PyYAML. Install with `pip install pyyaml`.") from e


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


def merge_csv_files(files: List[str], out_path: Path) -> None:
    """Concatenate CSVs, keeping header only from the first file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        header_written = False
        for fpath in files:
            with open(fpath, newline="", encoding="utf-8") as fin:
                reader = csv.reader(fin)
                rows = list(reader)
                if not rows:
                    continue
                if not header_written:
                    writer.writerows(rows)
                    header_written = True
                else:
                    writer.writerows(rows[1:])  # skip header from subsequent files


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    repo_root = Path(__file__).resolve().parents[1]

    ap.add_argument("--sweep_yaml", default=str(repo_root / "docs" / "full_retrain_sweep.yaml"))
    ap.add_argument("--config_file", default=str(repo_root / "configs" / "train_t5_pkm_warmup.yaml"))

    ap.add_argument("--with_pkm", action="store_true", help="Enable PKM memory layer")
    ap.add_argument("--force_routing_test", action="store_true", help="Apply group mask at test time (PKM only)")

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
    ap.add_argument("--master_port", type=int, default=int(os.environ.get("MASTER_PORT", "29500")))

    ap.add_argument("--local_ckpt_root", default=os.environ.get("LOCAL_CKPT_ROOT", f"/tmp/{os.environ.get('USER', 'user')}/memory_dev_ckpt"))
    ap.add_argument("--cleanup_ckpt", type=int, default=int(os.environ.get("CLEANUP_CKPT", "1")))
    ap.add_argument("--result_jsonl", default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--cuda_visible_devices", default=None)

    args = ap.parse_args()

    sweep = load_yaml(Path(args.sweep_yaml))
    train_cfg = sweep.get("train") if isinstance(sweep.get("train"), dict) else {}
    pkm_cfg = (sweep.get("pkm") or {}).get("t5_seq2seq") or {}

    lr = train_cfg.get("learning_rate", 1e-3)
    bs = train_cfg.get("batch_size", 512)
    epochs = int(args.epochs) if args.epochs is not None else int(train_cfg.get("epochs", 50))

    mode_tag = "pkm" if args.with_pkm else "no_pkm"
    run_tag = f"t5_full_retrain_{mode_tag}_lr{lr}_bs{bs}_ep{epochs}"
    log_subdir = f"sweep_t5_full_retrain_{mode_tag}"

    if args.result_jsonl is None:
        args.result_jsonl = f"./log/{args.dataset}/{log_subdir}/result.jsonl"
    result_jsonl = Path(args.result_jsonl)
    status_jsonl = result_jsonl.with_name("sweep_status.jsonl")

    run_ckpt_root = Path(args.local_ckpt_root) / "ckpt" / args.dataset / log_subdir / run_tag
    run_log_root = repo_root / "log" / args.dataset / log_subdir / run_tag
    test_log_dir = run_log_root / "test"
    test_log_dir.mkdir(parents=True, exist_ok=True)

    env_base = dict(os.environ)
    env_base["WANDB_MODE"] = "disabled"
    env_base["WANDB_DISABLED"] = "true"
    env_base["TRANSFORMERS_NO_TF"] = env_base.get("TRANSFORMERS_NO_TF", "1")
    env_base["USE_TF"] = env_base.get("USE_TF", "0")
    if args.cuda_visible_devices is not None:
        env_base["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    cfg_path = Path(args.config_file)
    if not cfg_path.is_file():
        raise FileNotFoundError(str(cfg_path))

    # PKM CLI overrides
    if args.with_pkm:
        pkm_overrides = [
            "pkm.t5_seq2seq.pk_is_enabled=true",
            f"pkm.t5_seq2seq.pk_encoder_layers={pkm_cfg.get('pk_encoder_layers', '')}",
            f"pkm.t5_seq2seq.pk_decoder_layers={pkm_cfg.get('pk_decoder_layers', '3')}",
            f"pkm.t5_seq2seq.pk_mem_n_keys={int(pkm_cfg.get('pk_mem_n_keys', 125))}",
            f"pkm.t5_seq2seq.pk_mem_knn={int(pkm_cfg.get('pk_mem_knn', 25))}",
            f"pkm.t5_seq2seq.pk_topk={int(pkm_cfg.get('pk_topk', 32))}",
            "pkm.t5_seq2seq.pk_warmup_epochs=0",
        ]
    else:
        pkm_overrides = ["pkm.t5_seq2seq.pk_is_enabled=false"]

    append_jsonl(status_jsonl, {
        "created_at": utc_now(), "status": "start",
        "run_tag": run_tag, "mode": mode_tag, "with_pkm": args.with_pkm,
    })

    params_json = run_log_root / "params.json"
    params_json.parent.mkdir(parents=True, exist_ok=True)
    params_json.write_text(json.dumps({
        "dataset": args.dataset,
        "mode": mode_tag,
        "run_tag": run_tag,
        "with_pkm": args.with_pkm,
        "train.learning_rate": lr,
        "train.batch_size": bs,
        "train.epochs": epochs,
        "BASE_MODEL": args.base_model,
        "TIME_RANGE": args.time_range,
    }, ensure_ascii=False, indent=2))

    with tempfile.TemporaryDirectory(prefix="full_retrain_") as tmpdir:
        failed = False

        for train_d in (0, 1, 2, 3):
            test_d = train_d + 1

            # Cumulative train data: D0 .. D{train_d}
            train_files = [
                os.path.join(args.data_root, f"D{d}", f"{args.dataset}_5_{args.time_range}.csv")
                for d in range(train_d + 1)
            ]
            merged_train = Path(tmpdir) / f"merged_D0_to_D{train_d}.csv"
            merge_csv_files(train_files, merged_train)

            cur_ckpt = run_ckpt_root / f"D{train_d}"
            cur_ckpt.mkdir(parents=True, exist_ok=True)
            train_log = run_log_root / "train" / f"{run_tag}_train_D0-D{train_d}.log"

            cmd_train = [
                "torchrun",
                "--nproc_per_node=1",
                f"--master_port={args.master_port + train_d}",
                "train.py",
                f"config={str(cfg_path)}",
                "strategy=t5_seq2seq",
                f"model.t5_seq2seq.base_model={args.base_model}",
                f"train.output_dir={str(cur_ckpt)}",
                f"train.logging_dir={str(run_log_root / f'D{train_d}' / 'runs')}",
                f"dataset.data_path={args.amazon_root}",
                f"dataset.name={args.dataset}",
                f"dataset.train_file={str(merged_train)}",
                f"dataset.valid_file={str(merged_train)}",
                f"dataset.test_file={str(merged_train)}",
                f"dataset.index_file={args.index_file}",
                f"train.batch_size={bs}",
                f"train.learning_rate={lr}",
                f"train.epochs={epochs}",
                f"train.weight_decay={args.wd}",
                "train.logging_step=1",
                "train.save_and_eval_strategy=epoch",
                f"train.model_max_length={args.model_max_length}",
            ] + pkm_overrides

            if args.dry_run:
                print(f"[DRY RUN] train D0..D{train_d}: {cmd_train[0]} ... (merged {len(train_files)} files)")
            else:
                rc = run_cmd(cmd_train, env=env_base, log_path=train_log)
                if rc != 0:
                    failed = True
                    append_jsonl(status_jsonl, {"created_at": utc_now(), "status": "failed", "phase": f"train_D0-D{train_d}", "returncode": rc})
                    break

            if failed:
                break

            # Test on D{test_d} per-group files
            group_dir = Path(args.data_root) / f"D{test_d}" / "groups"
            group_files = sorted(glob.glob(str(group_dir / "*.csv")))
            if not group_files:
                append_jsonl(status_jsonl, {"created_at": utc_now(), "status": "warn", "phase": f"test_D{test_d}", "warning": f"no group files in {group_dir}"})
            else:
                for group_file in group_files:
                    group_name = Path(group_file).stem
                    test_log = test_log_dir / f"{run_tag}_trainD0-D{train_d}_testD{test_d}_{group_name}.log"
                    cmd_test = [
                        "python", "test.py",
                        f"config={str(cfg_path)}",
                        "model.type=t5_seq2seq",
                        "global.gpu_id=0",
                        f"model.ckpt_path={str(cur_ckpt)}",
                        f"model.tokenizer_path={str(cur_ckpt)}",
                        f"model.base_model={args.base_model}",
                        f"dataset.name={args.dataset}",
                        f"dataset.data_path={args.amazon_root}",
                        f"dataset.train_file={str(merged_train)}",
                        f"dataset.valid_file={str(merged_train)}",
                        f"dataset.test_file={group_file}",
                        f"dataset.index_file={args.index_file}",
                        f"test.batch_size={args.test_batch_size}",
                        f"test.num_beams={args.num_beams}",
                        f"test.max_new_tokens={args.max_new_tokens}",
                        "test.filter_items=true",
                        "test.log_pkm_heatmap=false",
                        f"test.pk_force_routing={'true' if args.force_routing_test else 'false'}",
                        f"test.logging_dir={str(run_log_root / f'D{train_d}' / 'test' / 'runs' / f'testD{test_d}_{group_name}')}",
                    ] + pkm_overrides

                    if args.dry_run:
                        print(f"[DRY RUN] test D{test_d} {group_name}")
                    else:
                        rc2 = run_cmd(cmd_test, env=env_base, log_path=test_log)
                        if rc2 != 0:
                            failed = True
                            append_jsonl(status_jsonl, {"created_at": utc_now(), "status": "failed", "phase": f"test_{group_name}_D{test_d}", "returncode": rc2})
                            break

            if failed:
                break

            # Combined test for heatmap (all groups)
            combined_test_file = os.path.join(args.data_root, f"D{test_d}", f"{args.dataset}_5_{args.time_range}.csv")
            combined_name = f"testD{test_d}_allgroups"
            combined_log = test_log_dir / f"{run_tag}_trainD0-D{train_d}_testD{test_d}_{combined_name}.log"
            combined_tb_dir = run_log_root / f"D{train_d}" / "test" / "runs" / combined_name

            cmd_test_all = [
                "python", "test.py",
                f"config={str(cfg_path)}",
                "model.type=t5_seq2seq",
                "global.gpu_id=0",
                f"model.ckpt_path={str(cur_ckpt)}",
                f"model.tokenizer_path={str(cur_ckpt)}",
                f"model.base_model={args.base_model}",
                f"dataset.name={args.dataset}",
                f"dataset.data_path={args.amazon_root}",
                f"dataset.train_file={str(merged_train)}",
                f"dataset.valid_file={str(merged_train)}",
                f"dataset.test_file={combined_test_file}",
                f"dataset.index_file={args.index_file}",
                f"test.batch_size={args.test_batch_size}",
                f"test.num_beams={args.num_beams}",
                f"test.max_new_tokens={args.max_new_tokens}",
                "test.filter_items=true",
                "test.log_pkm_heatmap=true",
                f"test.pk_force_routing={'true' if args.force_routing_test else 'false'}",
                f"test.logging_dir={str(combined_tb_dir)}",
            ] + pkm_overrides

            if args.dry_run:
                print(f"[DRY RUN] test D{test_d} allgroups (heatmap)")
            else:
                rc3 = run_cmd(cmd_test_all, env=env_base, log_path=combined_log)
                if rc3 != 0:
                    failed = True
                    append_jsonl(status_jsonl, {"created_at": utc_now(), "status": "failed", "phase": f"test_allgroups_D{test_d}", "returncode": rc3})
                    break

        if not failed:
            out = subprocess.run(
                [
                    "python",
                    str(repo_root / "docs" / "write_result_jsonl.py"),
                    "--params_json", str(params_json),
                    "--run_tag", run_tag,
                    "--test_log_glob", str(test_log_dir / f"{run_tag}_trainD0-D*_testD*_*.log"),
                ],
                env=env_base,
                check=False,
                capture_output=True,
                text=True,
            )
            if out.returncode == 0:
                append_jsonl(result_jsonl, json.loads(out.stdout) if out.stdout.strip().startswith("{") else {"run_tag": run_tag, "raw": out.stdout})
            append_jsonl(status_jsonl, {"created_at": utc_now(), "status": "done", "run_tag": run_tag})

        if args.cleanup_ckpt == 1:
            try:
                subprocess.run(["rm", "-rf", str(run_ckpt_root)], check=False)
            except Exception:
                pass

        append_jsonl(status_jsonl, {
            "created_at": utc_now(),
            "status": "sweep_summary",
            "run_tag": run_tag,
            "mode": mode_tag,
            "failed": failed,
        })
        print(json.dumps({"run_tag": run_tag, "mode": mode_tag, "failed": failed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
