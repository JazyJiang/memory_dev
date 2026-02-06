import argparse
import glob
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple


_TEST_LINE_RE = re.compile(
    r"^\[Test\]: Precision:\s*(?P<p>[0-9.\-]+)\s+"
    r"Recall:\s*(?P<r>[0-9.\-]+)\s+"
    r"NDCG:\s*(?P<n>[0-9.\-]+)\s+"
    r"MRR:\s*(?P<m>[0-9.\-]+)\s*$"
)

_LOG_NAME_RE = re.compile(
    r".*_trainD(?P<train>\d+)_testD(?P<test>\d+)_(?P<group>.+)\.log$"
)


def _parse_list(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x) for x in s.split("-") if x != ""]


def _parse_one_log(path: Path) -> Optional[Dict[str, List[float]]]:
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

    m = _TEST_LINE_RE.match(test_line)
    if not m:
        return None

    precision = _parse_list(m.group("p"))
    recall = _parse_list(m.group("r"))
    ndcg = _parse_list(m.group("n"))
    mrr = _parse_list(m.group("m"))

    if not (len(precision) == len(recall) == len(ndcg) == len(mrr)):
        return None

    return {"precision": precision, "recall": recall, "ndcg": ndcg, "mrr": mrr}


def _mean_by_index(rows: List[List[float]]) -> List[float]:
    if not rows:
        return []
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        raise ValueError(f"Metric length mismatch across groups: {[len(r) for r in rows]}")
    return [mean([r[i] for r in rows]) for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params_json", required=True)
    ap.add_argument("--run_tag", required=True)

    # Old mode (per transition aggregation)
    ap.add_argument("--train_d", type=int, default=None)
    ap.add_argument("--test_d", type=int, default=None)

    # New mode (one line per run_tag)
    ap.add_argument("--test_log_glob", required=True)
    args = ap.parse_args()

    params = json.loads(Path(args.params_json).read_text())
    paths = [Path(p) for p in sorted(glob.glob(args.test_log_glob))]

    # -------------------------
    # Old mode: aggregate across groups for a single (train_d, test_d)
    # -------------------------
    if args.train_d is not None and args.test_d is not None:
        parsed: List[Tuple[str, Dict[str, List[float]]]] = []
        for p in paths:
            r = _parse_one_log(p)
            if r is None:
                continue
            parsed.append((str(p), r))

        out: Dict[str, object] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_tag": args.run_tag,
            "train_d": args.train_d,
            "test_d": args.test_d,
            "n_group_logs": len(paths),
            "n_group_parsed": len(parsed),
            "params": params,
            "group_logs": [lp for (lp, _) in parsed],
        }

        if parsed:
            precision = _mean_by_index([r["precision"] for _, r in parsed])
            recall = _mean_by_index([r["recall"] for _, r in parsed])
            ndcg = _mean_by_index([r["ndcg"] for _, r in parsed])
            mrr = _mean_by_index([r["mrr"] for _, r in parsed])

            ks = [5, 10, 20]
            metrics: Dict[str, float] = {}
            for i, k in enumerate(ks[: len(precision)]):
                metrics[f"precision@{k}"] = precision[i]
                metrics[f"recall@{k}"] = recall[i]
                metrics[f"ndcg@{k}"] = ndcg[i]
                metrics[f"mrr@{k}"] = mrr[i]
            out["metrics_mean"] = metrics
            out["metrics_mean_lists"] = {
                "precision": precision,
                "recall": recall,
                "ndcg": ndcg,
                "mrr": mrr,
                "ks": ks[: len(precision)],
            }

        print(json.dumps(out, ensure_ascii=False))
        return

    # -------------------------
    # New mode: one JSON per run_tag, nested by group then by transition
    # Structure:
    # {
    #   "params": {...},
    #   "test_results": {
    #     "group1": { "D0>D1": { "recall@5":..., "NDCG@5":... }, ... },
    #     "group2": { ... }
    #   }
    # }
    # -------------------------
    group_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    parsed_logs: List[str] = []
    skipped_logs: List[str] = []

    for p in paths:
        m = _LOG_NAME_RE.match(p.name)
        if not m:
            skipped_logs.append(str(p))
            continue

        train_d = int(m.group("train"))
        test_d = int(m.group("test"))
        group = m.group("group")

        r = _parse_one_log(p)
        if r is None:
            skipped_logs.append(str(p))
            continue

        transition = f"D{train_d}>D{test_d}"

        # Only output what you asked for: recall@{5,10,20} and NDCG@{5,10,20}
        ks = [5, 10, 20]
        recall = r["recall"]
        ndcg = r["ndcg"]

        metrics: Dict[str, float] = {}
        for i, k in enumerate(ks[: min(len(recall), len(ndcg))]):
            metrics[f"recall@{k}"] = recall[i]
            metrics[f"NDCG@{k}"] = ndcg[i]

        group_results.setdefault(group, {})[transition] = metrics
        parsed_logs.append(str(p))

    out2: Dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_tag": args.run_tag,
        "params": params,
        "n_test_logs": len(paths),
        "n_test_logs_parsed": len(parsed_logs),
        "n_test_logs_skipped": len(skipped_logs),
        "test_logs_parsed": parsed_logs,
        "test_logs_skipped": skipped_logs,
        "test_results": group_results,
    }

    print(json.dumps(out2, ensure_ascii=False))


if __name__ == "__main__":
    main()