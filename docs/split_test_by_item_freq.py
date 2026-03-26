"""
Split test CSV into hot/cold subsets based on cumulative item frequency
in D0~D{t-1} training data.

For each test period D{t} (t=1..4):
  1. Count how many times each item_id appears as target in D0..D{t-1} CSVs
  2. Split test rows by whether target item_id is hot (top 20%) or cold (bottom 20% + unseen)
  3. Write to D{t}/groups/test_item_hot.csv and D{t}/groups/test_item_cold.csv

Usage:
    python docs/split_test_by_item_freq.py \
        --data_root /path/to/data \
        --dataset Toys_and_Games_5_2016-10-2018-11 \
        --hot_pct 20 --cold_pct 20
"""

import argparse
import os
from collections import Counter
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True, help="Root data dir containing D0/ .. D4/")
    ap.add_argument("--dataset", required=True, help="Dataset filename stem, e.g. Toys_and_Games_5_2016-10-2018-11")
    ap.add_argument("--hot_pct", type=float, default=20, help="Top X%% frequency = hot (default 20)")
    ap.add_argument("--cold_pct", type=float, default=20, help="Bottom X%% frequency + unseen = cold (default 20)")
    args = ap.parse_args()

    data_root = Path(args.data_root)

    for test_d in range(1, 5):
        # 1. Accumulate item frequency from D0..D{t-1}
        item_freq = Counter()
        for train_d in range(test_d):
            csv_path = data_root / f"D{train_d}" / f"{args.dataset}.csv"
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found, skipping")
                continue
            df = pd.read_csv(csv_path)
            item_freq.update(df["item_id"].tolist())

        if not item_freq:
            print(f"D{test_d}: no training data found, skipping")
            continue

        # 2. Compute frequency thresholds
        freqs = sorted(item_freq.values())
        n = len(freqs)
        cold_threshold = freqs[max(0, int(n * args.cold_pct / 100) - 1)]
        hot_threshold = freqs[min(n - 1, int(n * (100 - args.hot_pct) / 100))]

        print(f"D{test_d}: {len(item_freq)} unique items in D0..D{test_d-1}, "
              f"cold_threshold={cold_threshold}, hot_threshold={hot_threshold}")

        # 3. Load test data — find the full test CSV for this period
        test_csv = data_root / f"D{test_d}" / f"{args.dataset}.csv"
        if not test_csv.exists():
            print(f"Warning: {test_csv} not found, skipping")
            continue
        test_df = pd.read_csv(test_csv)

        # Classify each row by target item frequency
        test_df["_item_freq"] = test_df["item_id"].map(lambda x: item_freq.get(x, 0))

        hot_mask = test_df["_item_freq"] >= hot_threshold
        cold_mask = test_df["_item_freq"] <= cold_threshold  # includes unseen (freq=0)

        hot_df = test_df[hot_mask].drop(columns=["_item_freq"])
        cold_df = test_df[cold_mask].drop(columns=["_item_freq"])

        # 4. Write to groups directory
        groups_dir = data_root / f"D{test_d}" / "groups"
        groups_dir.mkdir(parents=True, exist_ok=True)

        hot_path = groups_dir / "test_item_hot.csv"
        cold_path = groups_dir / "test_item_cold.csv"

        hot_df.to_csv(hot_path, index=False)
        cold_df.to_csv(cold_path, index=False)

        print(f"  hot: {len(hot_df)} rows -> {hot_path}")
        print(f"  cold: {len(cold_df)} rows -> {cold_path}")


if __name__ == "__main__":
    main()
