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


def plot_groups_stacked(records: List[Record], ks: List[int], dataset: str, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
    except Exception as e:
        raise RuntimeError(
            f"matplotlib is required for plotting but failed to import: {e}. "
            f"You can still use the CSV summary."
        )

    metric_names = ["precision", "recall", "ndcg", "mrr"]
    titles = {"precision": "Precision", "recall": "Recall", "ndcg": "NDCG", "mrr": "MRR"}

    groups = sorted({r.group for r in records})
    train_ds = sorted({r.train_d for r in records})
    test_ds = sorted({r.test_d for r in records})

    if not groups or not train_ds or not test_ds:
        raise RuntimeError("Not enough parsed records to plot per-group stacked figure.")

    by_key: Dict[Tuple[int, int, int], Record] = {(r.train_d, r.test_d, r.group): r for r in records}

    default_colors = plt.rcParams.get("axes.prop_cycle", None)
    if default_colors is not None:
        default_colors = default_colors.by_key().get("color", None)
    if not default_colors:
        default_colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    test_to_color: Dict[int, str] = {td: default_colors[i % len(default_colors)] for i, td in enumerate(test_ds)}

    ncols = len(ks)
    metric_rows = len(metric_names)

    k_header_rows = 1
    spacer_rows = metric_rows - 1
    label_rows = 1
    group_gap_rows = 1

    rows_per_group = k_header_rows + metric_rows + spacer_rows + label_rows + group_gap_rows
    nrows_total = len(groups) * rows_per_group

    height_ratios: List[float] = []
    for _ in groups:
        height_ratios.append(0.36)  # k header
        for m_i in range(metric_rows):
            height_ratios.append(1.0)
            if m_i != metric_rows - 1:
                height_ratios.append(0.38)
        height_ratios.append(0.34)  # label row (was 0.26)
        height_ratios.append(0.46)  # group gap

    per_k_header_in = 0.40
    per_metric_row_in = 1.35
    per_spacer_row_in = 0.36
    per_label_row_in = 0.44  # was 0.36
    per_group_gap_row_in = 0.55
    fig_height = (
        len(groups)
        * (
            per_k_header_in
            + metric_rows * per_metric_row_in
            + spacer_rows * per_spacer_row_in
            + per_label_row_in
            + per_group_gap_row_in
        )
        + 1.8
    )

    fig = plt.figure(figsize=(4.6 * ncols, fig_height))
    gs = fig.add_gridspec(
        nrows_total,
        ncols,
        height_ratios=height_ratios,
        hspace=0.12,
        wspace=0.25,
    )

    first_ax = None
    group_to_axes: Dict[int, List["plt.Axes"]] = {}

    for g_i, group in enumerate(groups):
        row_base = g_i * rows_per_group
        group_axes: List["plt.Axes"] = []

        header_row = row_base
        for k_i, k in enumerate(ks):
            ax_k = fig.add_subplot(gs[header_row, k_i])
            ax_k.axis("off")
            ax_k.text(0.5, 0.5, f"@{k}", ha="center", va="center", fontsize=10)
            group_axes.append(ax_k)

        metric_start_row = row_base + k_header_rows

        for m_i, metric in enumerate(metric_names):
            row = metric_start_row + (m_i * 2)

            for k_i, k in enumerate(ks):
                if first_ax is None:
                    ax = fig.add_subplot(gs[row, k_i])
                    first_ax = ax
                else:
                    ax = fig.add_subplot(gs[row, k_i], sharex=first_ax)

                for test_d in test_ds:
                    xs: List[int] = []
                    ys: List[float] = []
                    for train_d in train_ds:
                        rec = by_key.get((train_d, test_d, group))
                        if rec is None:
                            continue
                        xs.append(train_d)
                        ys.append(getattr(rec, metric)[k_i])
                    if xs:
                        ax.plot(
                            xs,
                            ys,
                            color=test_to_color[test_d],
                            marker="o",
                            linewidth=1.2,
                            markersize=3.2,
                            alpha=0.95,
                        )

                ax.grid(True, linewidth=0.3, alpha=0.5)
                ax.set_xticks(train_ds)
                ax.set_xticklabels([str(d) for d in train_ds])
                ax.tick_params(axis="both", labelsize=7)

                # 关键：每个 group 的最后一行显示横轴刻度，其余行隐藏
                if m_i == metric_rows - 1:
                    ax.tick_params(axis="x", labelbottom=True)
                else:
                    ax.tick_params(axis="x", labelbottom=False)

                # 删除这两段（否则会导致横轴全没字）
                # ax.label_outer()
                # if not (g_i == len(groups) - 1 and m_i == metric_rows - 1):
                #     ax.tick_params(axis="x", labelbottom=False)

                if k_i == 0:
                    ax.set_ylabel(titles[metric], fontsize=8)

                y_all: List[float] = []
                for test_d in test_ds:
                    for train_d in train_ds:
                        rec = by_key.get((train_d, test_d, group))
                        if rec is None:
                            continue
                        y_all.append(getattr(rec, metric)[k_i])
                if y_all:
                    y_min = min(y_all)
                    y_max = max(y_all)
                    span = y_max - y_min
                    pad = (span * 0.22) if span > 0 else (abs(y_max) * 0.10 + 1e-6)
                    ax.set_ylim(y_min - pad, y_max + pad)

                group_axes.append(ax)

        label_row = metric_start_row + (metric_rows + spacer_rows)
        ax_label = fig.add_subplot(gs[label_row, :])
        ax_label.axis("off")
        ax_label.text(
            0.5,
            0.15,  # was 0.35, move down to be farther from top box line
            f"group {group}",
            ha="center",
            va="center",
            fontsize=12,
        )

        gap_row = label_row + 1
        ax_gap = fig.add_subplot(gs[gap_row, :])
        ax_gap.axis("off")

        group_to_axes[group] = group_axes

    test_handles = [
        Line2D([0], [0], color=test_to_color[td], marker="o", linewidth=2, label=f"test D{td}")
        for td in test_ds
    ]
    # 底部留白收紧：让 legend/xlabel 更靠近图主体
    fig.legend(
        handles=test_handles,
        loc="lower center",
        ncol=min(len(test_handles), 6),
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),  # was 0.03
    )

    fig.suptitle(f"{dataset} grouped test — per-group stacked (boxed by group)", y=0.985)
    fig.supxlabel("Train Period (D)", y=0.062, fontsize=10)  # was 0.075

    fig.subplots_adjust(left=0.06, right=0.995, top=0.94, bottom=0.065)  # was 0.085

    pad = 0.004
    for group, axes_list in group_to_axes.items():
        x0 = min(ax.get_position().x0 for ax in axes_list)
        y0 = min(ax.get_position().y0 for ax in axes_list)
        x1 = max(ax.get_position().x1 for ax in axes_list)
        y1 = max(ax.get_position().y1 for ax in axes_list)

        rect = Rectangle(
            (x0 - pad, y0 - pad),
            (x1 - x0) + 2 * pad,
            (y1 - y0) + 2 * pad,
            transform=fig.transFigure,
            fill=False,
            linewidth=1.1,
            edgecolor="0.25",
            zorder=10,
            clip_on=False,
        )
        fig.add_artist(rect)

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

    per_group_stacked_path = out_dir / f"{args.dataset}_per_group_stacked.png"
    try:
        plot_groups_stacked(records, ks=ks, dataset=args.dataset, out_path=per_group_stacked_path)
    except RuntimeError as e:
        print(str(e))

    print(f"Wrote: {raw_csv}")
    if fig_path.exists():
        print(f"Wrote: {fig_path}")
    if per_group_stacked_path.exists():
        print(f"Wrote: {per_group_stacked_path}")


if __name__ == "__main__":
    main()