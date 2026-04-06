"""
Delta-set analysis: compare per-sample predictions across different history-truncation experiments.

Given prediction JSONL files from predict_per_sample.py, compute:
  S_ref  = hits in the reference model (e.g., trunc=2)
  S_cmp  = hits in the comparison model (e.g., trunc=5 or trunc=10)
  Delta  = S_cmp \\ S_ref   (what cmp gets right that ref misses)
  Delta_inv = S_ref \\ S_cmp  (what ref gets right that cmp misses)

Then analyze Delta to understand what early history contributes.

Usage:
    python analysis/delta_set_analysis.py \
        --ref  ./analysis_output/preds_h2_D0toD1_group1.jsonl \
        --cmp  ./analysis_output/preds_h5_D0toD1_group1.jsonl \
                ./analysis_output/preds_h10_D0toD1_group1.jsonl \
        --hit_k 10 \
        --output_dir ./analysis_output/delta_D0toD1_group1

Options:
    --ref      : reference prediction JSONL (e.g., the best trunc=2 model)
    --cmp      : one or more comparison prediction JSONLs (trunc=5, trunc=10, ...)
    --hit_k    : which hit@k to use for set membership (default: 10)
    --output_dir : where to write analysis results
    --no_semantic : skip semantic similarity analysis (faster)
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_preds(path: str) -> Dict[int, Dict]:
    """Load a prediction JSONL. Key = row_idx."""
    records: Dict[int, Dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records[int(obj["row_idx"])] = obj
    return records


def hit_set(records: Dict[int, Dict], hit_k: int) -> set:
    """Return set of row_idx where hit@hit_k is True."""
    key = f"hit@{hit_k}"
    return {idx for idx, r in records.items() if r.get(key, False)}


# ---------------------------------------------------------------------------
# Semantic similarity (word-level Jaccard, no extra deps)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> set:
    """Simple word-level tokenization."""
    import re
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_semantic_features(record: Dict) -> Dict:
    """
    For a single sample, compute:
      - sim_target_recent : Jaccard(target_title, last 2 history items)
      - sim_target_early  : Jaccard(target_title, all-but-last-2 history items = early history)
      - history_len       : number of history items used
      - early_history_len : number of early items (history_len - 2, floored at 0)
    """
    target_tokens = tokenize(record.get("target_title", ""))
    titles = record.get("history_titles", [])
    h_len = len(titles)

    recent_titles = titles[-2:] if h_len >= 2 else titles
    early_titles = titles[:-2] if h_len > 2 else []

    recent_tokens = tokenize(" ".join(recent_titles))
    early_tokens = tokenize(" ".join(early_titles))

    return {
        "sim_target_recent": jaccard(target_tokens, recent_tokens),
        "sim_target_early": jaccard(target_tokens, early_tokens),
        "history_len": h_len,
        "early_history_len": len(early_titles),
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(v: float, decimals: int = 4) -> str:
    if math.isnan(v):
        return "nan"
    return f"{v:.{decimals}f}"


def group_distribution(indices: set, records: Dict[int, Dict]) -> Dict[int, int]:
    """Count how many samples in `indices` belong to each user group."""
    counts: Dict[int, int] = defaultdict(int)
    for idx in indices:
        g = records[idx].get("group_id", -1)
        counts[int(g)] += 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse_delta(
    delta: set,
    delta_inv: set,
    shared_hits: set,
    shared_miss: set,
    ref_records: Dict[int, Dict],
    cmp_records: Dict[int, Dict],
    cmp_label: str,
    hit_k: int,
    output_dir: Path,
) -> Dict:
    """
    Analyse a single (ref, cmp) delta pair.
    Returns a summary dict and writes detailed JSONL to output_dir.
    """
    all_row_ids = set(ref_records.keys()) & set(cmp_records.keys())
    n_total = len(all_row_ids)

    summary = {
        "cmp_label": cmp_label,
        "hit_k": hit_k,
        "n_total_aligned": n_total,
        "n_shared_hits": len(shared_hits),
        "n_shared_miss": len(shared_miss),
        "n_delta_cmp_wins": len(delta),        # cmp hits, ref misses
        "n_delta_ref_wins": len(delta_inv),    # ref hits, cmp misses
        "delta_cmp_pct": len(delta) / n_total if n_total else 0,
        "delta_ref_pct": len(delta_inv) / n_total if n_total else 0,
    }

    # Group distributions
    summary["delta_cmp_group_dist"] = group_distribution(delta, cmp_records)
    summary["delta_ref_group_dist"] = group_distribution(delta_inv, ref_records)
    summary["shared_hits_group_dist"] = group_distribution(shared_hits, ref_records)

    # Semantic similarity analysis
    def sem_stats(indices: set, records: Dict[int, Dict], tag: str) -> Dict:
        feats = [compute_semantic_features(records[i]) for i in indices if i in records]
        if not feats:
            return {}
        return {
            f"{tag}_sim_target_recent_mean": mean([f["sim_target_recent"] for f in feats]),
            f"{tag}_sim_target_early_mean": mean([f["sim_target_early"] for f in feats]),
            f"{tag}_history_len_mean": mean([f["history_len"] for f in feats]),
            f"{tag}_early_history_len_mean": mean([f["early_history_len"] for f in feats]),
            f"{tag}_n": len(feats),
        }

    # Compare Delta (cmp wins) vs shared_hits (both win) vs delta_ref_wins (ref wins)
    summary.update(sem_stats(delta, cmp_records, "delta_cmp"))
    summary.update(sem_stats(delta_inv, ref_records, "delta_ref"))
    summary.update(sem_stats(shared_hits, ref_records, "shared_hits"))

    # Write delta JSONL for manual inspection
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_jsonl = output_dir / f"delta_cmp_wins_{cmp_label}.jsonl"
    with open(delta_jsonl, "w") as f:
        for idx in sorted(delta):
            r = cmp_records.get(idx, {})
            sem = compute_semantic_features(r)
            f.write(json.dumps({**r, **sem}, ensure_ascii=False) + "\n")

    delta_inv_jsonl = output_dir / f"delta_ref_wins_vs_{cmp_label}.jsonl"
    with open(delta_inv_jsonl, "w") as f:
        for idx in sorted(delta_inv):
            r = ref_records.get(idx, {})
            sem = compute_semantic_features(r)
            f.write(json.dumps({**r, **sem}, ensure_ascii=False) + "\n")

    return summary


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def print_summary(summary: Dict):
    cmp = summary["cmp_label"]
    k = summary["hit_k"]
    n = summary["n_total_aligned"]
    print(f"\n{'='*60}")
    print(f"  Delta-set analysis: ref vs {cmp}  (hit@{k})")
    print(f"{'='*60}")
    print(f"  Aligned samples         : {n}")
    print(f"  Both hit                : {summary['n_shared_hits']} ({summary['n_shared_hits']/n:.1%})")
    print(f"  Both miss               : {summary['n_shared_miss']} ({summary['n_shared_miss']/n:.1%})")
    print(f"  cmp wins, ref misses (Δ): {summary['n_delta_cmp_wins']} ({summary['delta_cmp_pct']:.1%})")
    print(f"  ref wins, cmp misses    : {summary['n_delta_ref_wins']} ({summary['delta_ref_pct']:.1%})")

    print(f"\n  Group distribution of Δ (cmp wins):")
    for g, cnt in summary.get("delta_cmp_group_dist", {}).items():
        print(f"    group{g}: {cnt}")

    print(f"\n  Semantic similarity (target ↔ history):")
    print(f"  {'Set':<20} {'sim_recent':>12} {'sim_early':>12} {'his_len':>8} {'early_len':>10} {'n':>6}")
    for tag, label in [("delta_cmp", f"Δ cmp-wins"), ("delta_ref", f"Δ ref-wins"), ("shared_hits", "shared hits")]:
        sr = summary.get(f"{tag}_sim_target_recent_mean", float("nan"))
        se = summary.get(f"{tag}_sim_target_early_mean", float("nan"))
        hl = summary.get(f"{tag}_history_len_mean", float("nan"))
        el = summary.get(f"{tag}_early_history_len_mean", float("nan"))
        nn = summary.get(f"{tag}_n", 0)
        print(f"  {label:<20} {fmt(sr):>12} {fmt(se):>12} {fmt(hl):>8} {fmt(el):>10} {nn:>6}")

    print()
    print("  Key question: Does Δ (cmp wins) have higher sim_target_early than shared_hits?")
    sr_delta = summary.get("delta_cmp_sim_target_early_mean", float("nan"))
    sr_shared = summary.get("shared_hits_sim_target_early_mean", float("nan"))
    if not math.isnan(sr_delta) and not math.isnan(sr_shared) and sr_shared > 0:
        direction = "YES (early history helps in Δ)" if sr_delta > sr_shared else "NO (early history not decisive)"
        print(f"  → {direction}  (Δ early sim={fmt(sr_delta)}, shared early sim={fmt(sr_shared)})")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="Reference prediction JSONL (e.g., trunc=2 model)")
    ap.add_argument("--cmp", nargs="+", required=True, help="Comparison prediction JSONLs (trunc=5, trunc=10, ...)")
    ap.add_argument("--hit_k", type=int, default=10, help="Which hit@k to use for set membership (default: 10)")
    ap.add_argument("--output_dir", default="./analysis_output/delta", help="Output directory")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading reference predictions: {args.ref}")
    ref_records = load_preds(args.ref)
    ref_hits = hit_set(ref_records, args.hit_k)
    ref_label = Path(args.ref).stem
    print(f"  {len(ref_records)} samples, {len(ref_hits)} hits ({len(ref_hits)/len(ref_records):.1%})")

    all_summaries = []

    for cmp_path in args.cmp:
        cmp_label = Path(cmp_path).stem
        print(f"\nLoading comparison predictions: {cmp_path} (label={cmp_label})")
        cmp_records = load_preds(cmp_path)
        cmp_hits = hit_set(cmp_records, args.hit_k)
        print(f"  {len(cmp_records)} samples, {len(cmp_hits)} hits ({len(cmp_hits)/len(cmp_records):.1%})")

        # Align by row_idx intersection
        common_ids = set(ref_records.keys()) & set(cmp_records.keys())
        if len(common_ids) < len(ref_records):
            print(f"  Warning: only {len(common_ids)}/{len(ref_records)} samples aligned by row_idx")

        ref_hits_common = ref_hits & common_ids
        cmp_hits_common = cmp_hits & common_ids

        delta = cmp_hits_common - ref_hits_common        # cmp wins, ref misses
        delta_inv = ref_hits_common - cmp_hits_common    # ref wins, cmp misses
        shared_hits = ref_hits_common & cmp_hits_common
        shared_miss = common_ids - ref_hits_common - cmp_hits_common

        summary = analyse_delta(
            delta=delta,
            delta_inv=delta_inv,
            shared_hits=shared_hits,
            shared_miss=shared_miss,
            ref_records={k: ref_records[k] for k in common_ids},
            cmp_records={k: cmp_records[k] for k in common_ids},
            cmp_label=cmp_label,
            hit_k=args.hit_k,
            output_dir=output_dir,
        )
        summary["ref_label"] = ref_label
        all_summaries.append(summary)
        print_summary(summary)

    # Write combined summary JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"Full summary written to {summary_path}")


if __name__ == "__main__":
    main()
