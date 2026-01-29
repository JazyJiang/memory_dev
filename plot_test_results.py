import argparse
import ast
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Record:
    train_d: int
    test_d: int
    group: int
    precision: List[float]
    recall: List[float]
    ndcg: List[float]
    mrr: List[float]
    log_path: str


_FILE_RE = re.compile(
    r"TIGER-D(?P<train>\d+)-test-D(?P<test>\d+)-test_user_group(?P<group>\d+).*\.log$"
)

_TEST_LINE_RE = re.compile(
    r"^\[Test\]: Precision:\s*(?P<p>[0-9.\-]+)\s+"
    r"Recall:\s*(?P<r>[0-9.\-]+)\s+"
    r"NDCG:\s*(?P<n>[0-9.\-]+)\s+"
    r"MRR:\s*(?P<m>[0-9.\-]+)\s*$"
)


def _parse_list(s: str) -> List[float]:
    parts = [p for p in s.strip().split("-") if p != ""]
    return [float(p) for p in parts]


def parse_one_log(path: Path, ks: List[int]) -> Optional[Record]:
    m = _FILE_RE.search(path.name)
    if not m:
        return None

    train_d = int(m.group("train"))
    test_d = int(m.group("test"))
    group = int(m.group("group"))

    try:
        lines = path.read_text(errors="ignore").splitlines()
    except Exception:
        return None

    test_line = None
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("[Test]: Precision:"):
            test_line = line
            break
    if not test_line:
        return None

    mm = _TEST_LINE_RE.match(test_line)
    if not mm:
        return None

    precision = _parse_list(mm.group("p"))
    recall = _parse_list(mm.group("r"))
    ndcg = _parse_list(mm.group("n"))
    mrr = _parse_list(mm.group("m"))

    exp = len(ks)
    if not (len(precision) == len(recall) == len(ndcg) == len(mrr) == exp):
        raise ValueError(
            f"Metric length mismatch in {path.name}: got "
            f"P={len(precision)}, R={len(recall)}, N={len(ndcg)}, M={len(mrr)}; expected {exp} (ks={ks})"
        )

    return Record(
        train_d=train_d,
        test_d=test_d,
        group=group,
        precision=precision,
        recall=recall,
        ndcg=ndcg,
        mrr=mrr,
        log_path=str(path),
    )


def load_records(log_dir: Path, ks: List[int]) -> List[Record]:
    records: List[Record] = []
    for p in sorted(log_dir.glob("*.log")):
        rec = parse_one_log(p, ks=ks)
        if rec is not None:
            records.append(rec)
    return records


def mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def std(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    mu = mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def aggregate(records: List[Record], ks: List[int]) -> Dict[Tuple[int, int], Dict[str, float]]:
    by_key: Dict[Tuple[int, int], List[Record]] = {}
    for r in records:
        by_key.setdefault((r.train_d, r.test_d), []).append(r)

    metrics = ["precision", "recall", "ndcg", "mrr"]

    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    for key, rs in by_key.items():
        row: Dict[str, float] = {"n_groups": float(len(rs))}
        for metric in metrics:
            for idx, k in enumerate(ks):
                xs = [getattr(r, metric)[idx] for r in rs]
                row[f"{metric}_{k}_mean"] = mean(xs)
                row[f"{metric}_{k}_std"] = std(xs)
        out[key] = row
    return out


def write_csv(records: List[Record], ks: List[int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric_cols: List[str] = []
    for metric in ["precision", "recall", "ndcg", "mrr"]:
        for k in ks:
            metric_cols.append(f"{metric}_{k}")

    fields = ["train_d", "test_d", "group"] + metric_cols + ["log_path"]

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = {
                "train_d": r.train_d,
                "test_d": r.test_d,
                "group": r.group,
                "log_path": r.log_path,
            }
            for metric in ["precision", "recall", "ndcg", "mrr"]:
                vals = getattr(r, metric)
                for idx, k in enumerate(ks):
                    row[f"{metric}_{k}"] = vals[idx]
            w.writerow(row)


def plot(agg: Dict[Tuple[int, int], Dict[str, float]], ks: List[int], dataset: str, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            f"matplotlib is required for plotting but failed to import: {e}. "
            f"You can still use the CSV summary."
        )

    train_ds = sorted({k[0] for k in agg.keys()})
    test_ds = sorted({k[1] for k in agg.keys()})

    metric_names = ["precision", "recall", "ndcg", "mrr"]
    titles = {"precision": "Precision", "recall": "Recall", "ndcg": "NDCG", "mrr": "MRR"}

    nrows = len(metric_names)
    ncols = len(ks)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.2 * nrows), squeeze=False)

    for r_i, metric in enumerate(metric_names):
        for c_i, k in enumerate(ks):
            ax = axes[r_i][c_i]
            mean_k = f"{metric}_{k}_mean"
            std_k = f"{metric}_{k}_std"

            for train_d in train_ds:
                xs: List[int] = []
                ys: List[float] = []
                es: List[float] = []
                for test_d in test_ds:
                    row = agg.get((train_d, test_d))
                    if not row:
                        continue
                    xs.append(test_d)
                    ys.append(row[mean_k])
                    es.append(row[std_k])
                if xs:
                    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, linewidth=1.2, label=f"train D{train_d}")

            ax.set_title(f"{titles[metric]}@{k}")
            ax.grid(True, linewidth=0.3, alpha=0.5)
            ax.set_xticks(sorted(set(test_ds)))

            if r_i == nrows - 1:
                ax.set_xlabel("Test Period (D)")
            if c_i == 0:
                ax.set_ylabel("Mean±std over user groups")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=False)

    fig.suptitle(f"{dataset} grouped test (mean±std over user groups)", y=0.995)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def infer_ks_from_topn_source(path: Path) -> Optional[List[int]]:
    if not path.exists() or not path.is_file():
        return None

    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return None

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_id = getattr(node.func, "id", None)
        func_attr = getattr(node.func, "attr", None)
        if func_id != "computeTopNAccuracy" and func_attr != "computeTopNAccuracy":
            continue

        for kw in node.keywords:
            if kw.arg != "topN":
                continue
            try:
                val = ast.literal_eval(kw.value)
            except Exception:
                return None
            if not isinstance(val, (list, tuple)):
                return None
            ks = [int(x) for x in val]
            if not ks or any(k <= 0 for k in ks):
                return None
            return ks

    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="dataset")
    ap.add_argument(
        "--ks",
        type=str,
        default=None,
        help="Comma-separated list like 5,10,20. If omitted, tries to infer from --topn_source.",
    )
    ap.add_argument(
        "--topn_source",
        type=str,
        default=None,
        help="Python file to infer topN=[...] from computeTopNAccuracy(..., topN=[...]). Defaults to ./test.py next to this script.",
    )
    args = ap.parse_args()

    if args.ks is not None:
        ks = [int(x) for x in args.ks.split(",") if x.strip() != ""]
        if not ks:
            raise SystemExit("--ks must be a comma-separated list of integers, e.g. 5,10,20")
    else:
        topn_source = (
            Path(args.topn_source)
            if args.topn_source
            else (Path(__file__).resolve().parent / "test.py")
        )
        ks = infer_ks_from_topn_source(topn_source) or [5, 10, 20]

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)

    records = load_records(log_dir, ks=ks)
    if not records:
        raise SystemExit(f"No parsable test logs found in: {log_dir}")

    raw_csv = out_dir / f"{args.dataset}_test_logs_parsed.csv"
    write_csv(records, ks=ks, out_path=raw_csv)

    agg = aggregate(records, ks=ks)
    fig_path = out_dir / f"{args.dataset}_test_metrics.png"
    try:
        plot(agg, ks=ks, dataset=args.dataset, out_path=fig_path)
    except RuntimeError as e:
        print(str(e))

    print(f"Wrote: {raw_csv}")
    if fig_path.exists():
        print(f"Wrote: {fig_path}")


if __name__ == "__main__":
    main()