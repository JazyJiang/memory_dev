"""
Dataset statistics: per-group user counts and prev interaction counts.

Output for each period D{t}:
  - Train: unique users per group, total interactions per group
  - Test:  unique users per group (from groups/ files), total rows per group
  - D1-D4: prev_interactions stats per group (median, mean, min, max)
  - D0 has no prev info, groups are assigned by self-frequency

Usage:
  python docs/dataset_stats.py
  python docs/dataset_stats.py --data_root /path/to/data --dataset Toys_and_Games_5_2016-10-2018-11
"""
import argparse
import ast
import glob
import json
import os

import pandas as pd


def count_history_length(history_str):
    try:
        items = ast.literal_eval(history_str)
        return len(items) if isinstance(items, list) else 0
    except Exception:
        return 0


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def stats_for_period(t: int, data_root: str, dataset: str):
    period_dir = os.path.join(data_root, f"D{t}")
    main_csv = os.path.join(period_dir, f"{dataset}.csv")
    group_map_path = os.path.join(period_dir, "user_group_map.json")
    groups_dir = os.path.join(period_dir, "groups")

    print_section(f"D{t}")

    if not os.path.isfile(main_csv):
        print(f"  [SKIP] Main CSV not found: {main_csv}")
        return

    df = pd.read_csv(main_csv)

    # ── Group map ──────────────────────────────────────────────────────
    if os.path.isfile(group_map_path):
        with open(group_map_path) as f:
            raw_map = json.load(f)
        # keys may be str or int
        group_map = {str(k): int(v) for k, v in raw_map.items()}
        df["group_idx"] = df["user_id"].astype(str).map(group_map)
    else:
        print("  [WARN] user_group_map.json not found; group info unavailable")
        df["group_idx"] = None

    # ── Train stats ────────────────────────────────────────────────────
    print(f"\n  [Train]  Total rows: {len(df):,}  |  Unique users: {df['user_id'].nunique():,}")
    if df["group_idx"].notna().any():
        train_grp = (
            df.groupby("group_idx")
            .agg(users=("user_id", "nunique"), interactions=("user_id", "count"))
            .reset_index()
            .sort_values("group_idx")
        )
        print(f"  {'Group':<8} {'Users':>10} {'Interactions':>15}")
        print(f"  {'-'*35}")
        for _, row in train_grp.iterrows():
            print(f"  {int(row['group_idx']):<8} {int(row['users']):>10,} {int(row['interactions']):>15,}")

    # ── Prev interactions per group (D1-D4 only) ──────────────────────
    if t > 0:
        prev_csv = os.path.join(data_root, f"D{t-1}", f"{dataset}.csv")
        if os.path.isfile(prev_csv):
            prev_df = pd.read_csv(prev_csv)
            prev_df["history_len"] = prev_df["history_item_id"].apply(count_history_length)
            user_pop = (
                prev_df.groupby("user_id")["history_len"]
                .max()
                .rename("prev_interactions")
                .reset_index()
            )
            curr_users = pd.DataFrame({"user_id": df["user_id"].unique()})
            curr_users["user_id_str"] = curr_users["user_id"].astype(str)
            user_df = curr_users.merge(user_pop, on="user_id", how="left")
            user_df["prev_interactions"] = user_df["prev_interactions"].fillna(0)
            if df["group_idx"].notna().any():
                user_df["group_idx"] = user_df["user_id"].astype(str).map(group_map)
                print(f"\n  [Prev Interactions from D{t-1}]  (per group, unique users)")
                print(f"  {'Group':<8} {'Users':>8} {'Median':>8} {'Mean':>8} {'Min':>6} {'Max':>6}")
                print(f"  {'-'*46}")
                for g, gdf in user_df.groupby("group_idx"):
                    pi = gdf["prev_interactions"]
                    print(f"  {int(g):<8} {len(gdf):>8,} {pi.median():>8.1f} {pi.mean():>8.1f} {int(pi.min()):>6} {int(pi.max()):>6}")
        else:
            print(f"\n  [WARN] Prev CSV not found: {prev_csv}")
    else:
        print(f"\n  [Prev Interactions]  D0 has no prev period — groups assigned by self-frequency")

    # ── Test stats (per-group files) ───────────────────────────────────
    group_files = sorted(glob.glob(os.path.join(groups_dir, "*.csv")))
    if group_files:
        print(f"\n  [Test]   Group files found: {len(group_files)}")
        print(f"  {'Group file':<30} {'Users':>10} {'Rows':>10}")
        print(f"  {'-'*52}")
        for gf in group_files:
            gdf = pd.read_csv(gf)
            fname = os.path.basename(gf)
            print(f"  {fname:<30} {gdf['user_id'].nunique():>10,} {len(gdf):>10,}")
    else:
        print(f"\n  [Test]   No group files found in: {groups_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default=os.environ.get(
        "DATA_ROOT",
        "/mlx_devbox/users/zhuosong.jiang/playground/memory_dev/data"
    ))
    ap.add_argument("--dataset", default=os.environ.get(
        "DATASET_FILE", "Toys_and_Games_5_2016-10-2018-11"
    ))
    ap.add_argument("--periods", default="0,1,2,3,4")
    args = ap.parse_args()

    periods = [int(x) for x in args.periods.split(",")]
    print(f"Data root : {args.data_root}")
    print(f"Dataset   : {args.dataset}")
    print(f"Periods   : D{periods[0]} .. D{periods[-1]}")

    for t in periods:
        stats_for_period(t, args.data_root, args.dataset)

    print(f"\n{'='*60}")
    print("  Done")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
