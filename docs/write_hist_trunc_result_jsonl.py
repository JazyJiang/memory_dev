"""
Collect history-truncation test logs into a single JSONL record.

Log filename pattern expected:
    {run_tag}_trainD{train}_testD{test}_{group}_hlen{hist_len}.log

Output (one JSON line per run_tag):
{
  "run_tag": "...",
  "params": {...},
  "hist_trunc_results": {
    "test_user_group1": {
      "D0>D1": {
        "1":  {"recall@5": ..., "recall@10": ..., "recall@20": ..., "NDCG@20": ...},
        "5":  {...},
        "10": {...}
      }
    },
    ...
  }
}
"""

import argparse
import glob
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_TEST_LINE_RE = re.compile(
    r"^\[Test\]: Precision:\s*(?P<p>[0-9.\-]+)\s+"
    r"Recall:\s*(?P<r>[0-9.\-]+)\s+"
    r"NDCG:\s*(?P<n>[0-9.\-]+)\s+"
    r"MRR:\s*(?P<m>[0-9.\-]+)\s*$"
)

# Matches: {anything}_trainD{d}_testD{d}_{group}_hlen{len}.log
_LOG_NAME_RE = re.compile(
    r".*_trainD(?P<train>\d+)_testD(?P<test>\d+)_(?P<group>.+)_hlen(?P<hlen>\d+)\.log$"
)


def _parse_list(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x) for x in s.split("-") if x != ""]


def _parse_log(path: Path) -> Optional[Dict[str, List[float]]]:
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except Exception:
        return None

    for line in reversed(lines):
        line = line.strip()
        if line.startswith("[Test]: Precision:"):
            m = _TEST_LINE_RE.match(line)
            if not m:
                return None
            return {
                "precision": _parse_list(m.group("p")),
                "recall": _parse_list(m.group("r")),
                "ndcg": _parse_list(m.group("n")),
                "mrr": _parse_list(m.group("m")),
            }
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params_json", required=True)
    ap.add_argument("--run_tag", required=True)
    ap.add_argument("--log_glob", required=True, help="Glob for hist-trunc log files")
    args = ap.parse_args()

    params = json.loads(Path(args.params_json).read_text())
    paths = [Path(p) for p in sorted(glob.glob(args.log_glob))]

    KS = [5, 10, 20]

    # results[group][transition][hist_len_str] = {metric: value}
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
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
        hlen = m.group("hlen")
        transition = f"D{train_d}>D{test_d}"

        r = _parse_log(p)
        if r is None:
            skipped_logs.append(str(p))
            continue

        metrics: Dict[str, float] = {}
        for i, k in enumerate(KS[: min(len(r["recall"]), len(r["ndcg"]))]):
            metrics[f"recall@{k}"] = r["recall"][i]
            metrics[f"NDCG@{k}"] = r["ndcg"][i]

        results.setdefault(group, {}).setdefault(transition, {})[hlen] = metrics
        parsed_logs.append(str(p))

    out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_tag": args.run_tag,
        "params": params,
        "n_logs": len(paths),
        "n_parsed": len(parsed_logs),
        "n_skipped": len(skipped_logs),
        "skipped_logs": skipped_logs,
        "hist_trunc_results": results,
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
