"""
Dataset statistics v2: deeper analysis for PKM forced-router diagnosis.

新增模块（在原有统计基础上）:
  A. Cross-period group transition matrix — 用户group标签在相邻period间的迁移矩阵
     → 若对角线不占主导，说明group assignment不稳定，forced routing假设失效
  B. History length distribution at test time per group
     → 测试时各group的实际history长度，验证"cold group"是否真的history短
  C. New vs returning users per group per period
     → 各group中真正首次出现(new)的用户占比
  D. Item overlap across groups (Jaccard similarity)
     → 各group消费的item集合重叠度，高重叠=行为差异小=难以专门化keys
  E. Prev-interactions distribution shape per group
     → 各group的prev_interactions分位点，验证group边界是否有意义
  F. Target item popularity per group (长尾分析)
     → 核心问题：group0（最活跃用户）的next item是否更多是冷门/长尾item？
     → 用全局item交互频次定义popularity，分析各group的target item popularity分布
     → 若group0的target items更偏长尾 → recall@20低是item难度导致，而非模型对warm user表征差

Usage:
  python docs/dataset_stats_v2.py
  python docs/dataset_stats_v2.py --data_root /path --dataset Toys_and_Games_5_2016-10-2018-11
  python docs/dataset_stats_v2.py --modules F        # 只跑长尾分析
  python docs/dataset_stats_v2.py --modules A,B,C,D,E,F
"""
import argparse
import ast
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


# ─────────────────────────── helpers ────────────────────────────────────────

def count_history_length(history_str):
    try:
        items = ast.literal_eval(history_str)
        return len(items) if isinstance(items, list) else 0
    except Exception:
        return 0


def parse_history_items(history_str):
    """Return list of item ids from history string."""
    try:
        items = ast.literal_eval(history_str)
        return items if isinstance(items, list) else []
    except Exception:
        return []


def load_group_map(period_dir):
    """Load user_group_map.json → {str(user_id): int(group_idx)} or None."""
    path = os.path.join(period_dir, "user_group_map.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    return {str(k): int(v) for k, v in raw.items()}


def load_period_df(data_root, t, dataset):
    path = os.path.join(data_root, f"D{t}", f"{dataset}.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_sub(title):
    print(f"\n  ── {title} ──")


# ─────────────────────── original stats (preserved) ────────────────────────

def stats_for_period_original(t, data_root, dataset):
    period_dir = os.path.join(data_root, f"D{t}")
    main_csv = os.path.join(period_dir, f"{dataset}.csv")
    groups_dir = os.path.join(period_dir, "groups")

    print_section(f"D{t}  [original stats]")

    if not os.path.isfile(main_csv):
        print(f"  [SKIP] CSV not found: {main_csv}")
        return

    df = pd.read_csv(main_csv)
    group_map = load_group_map(period_dir)
    if group_map:
        df["group_idx"] = df["user_id"].astype(str).map(group_map)
    else:
        print("  [WARN] user_group_map.json not found")
        df["group_idx"] = None

    print(f"\n  [Train]  Total rows: {len(df):,}  |  Unique users: {df['user_id'].nunique():,}")
    if df["group_idx"].notna().any():
        train_grp = (
            df.groupby("group_idx")
            .agg(users=("user_id", "nunique"), interactions=("user_id", "count"))
            .reset_index().sort_values("group_idx")
        )
        print(f"  {'Group':<8} {'Users':>10} {'Interactions':>15}")
        print(f"  {'-'*35}")
        for _, row in train_grp.iterrows():
            print(f"  {int(row['group_idx']):<8} {int(row['users']):>10,} {int(row['interactions']):>15,}")

    if t > 0:
        prev_csv = os.path.join(data_root, f"D{t-1}", f"{dataset}.csv")
        if os.path.isfile(prev_csv):
            prev_df = pd.read_csv(prev_csv)
            prev_df["history_len"] = prev_df["history_item_id"].apply(count_history_length)
            user_pop = prev_df.groupby("user_id")["history_len"].max().rename("prev_interactions").reset_index()
            curr_users = pd.DataFrame({"user_id": df["user_id"].unique()})
            user_df = curr_users.merge(user_pop, on="user_id", how="left")
            user_df["prev_interactions"] = user_df["prev_interactions"].fillna(0)
            if group_map:
                user_df["group_idx"] = user_df["user_id"].astype(str).map(group_map)
                print(f"\n  [Prev Interactions from D{t-1}]")
                print(f"  {'Group':<8} {'Users':>8} {'Median':>8} {'Mean':>8} {'Min':>6} {'Max':>6}")
                print(f"  {'-'*46}")
                for g, gdf in user_df.groupby("group_idx"):
                    pi = gdf["prev_interactions"]
                    print(f"  {int(g):<8} {len(gdf):>8,} {pi.median():>8.1f} {pi.mean():>8.1f} {int(pi.min()):>6} {int(pi.max()):>6}")
    else:
        print(f"\n  [Prev]  D0 has no prev period — groups assigned by self-frequency")

    group_files = sorted(glob.glob(os.path.join(groups_dir, "*.csv")))
    if group_files:
        print(f"\n  [Test]  Group files: {len(group_files)}")
        print(f"  {'File':<30} {'Users':>8} {'Interactions':>14}")
        print(f"  {'-'*54}")
        for gf in group_files:
            gdf = pd.read_csv(gf)
            print(f"  {os.path.basename(gf):<30} {gdf['user_id'].nunique():>8,} {len(gdf):>14,}")


# ─────────── Module A: cross-period group transition matrix ─────────────────

def module_A_group_transition(data_root, dataset, periods):
    """
    对相邻 period (D{t-1}, D{t})，找到两期都出现的用户，
    计算其 group_idx 的迁移矩阵。

    核心问题：group assignment 是否稳定？
    - 若对角线占主导 → 用户倾向于维持相同group → forced routing 有意义
    - 若矩阵较均匀 → group assignment 基本随机 → forced routing 无意义
    """
    print_section("Module A: Cross-Period Group Transition Matrix")
    print("  (group 0 = most active, group 4 = least active in training map)")
    print("  Rows = group in D{t-1}, Cols = group in D{t}")

    for t in periods:
        if t == 0:
            continue
        gmap_prev = load_group_map(os.path.join(data_root, f"D{t-1}"))
        gmap_curr = load_group_map(os.path.join(data_root, f"D{t}"))
        if gmap_prev is None or gmap_curr is None:
            print(f"\n  D{t-1}→D{t}: [SKIP] missing user_group_map.json")
            continue

        df_curr = load_period_df(data_root, t, dataset)
        if df_curr is None:
            continue
        curr_users = df_curr["user_id"].astype(str).unique()

        # Only users appearing in both periods
        common_users = [u for u in curr_users if u in gmap_prev and u in gmap_curr]
        if not common_users:
            print(f"\n  D{t-1}→D{t}: no common users")
            continue

        groups_prev = [gmap_prev[u] for u in common_users]
        groups_curr = [gmap_curr[u] for u in common_users]

        n_groups = 5
        matrix = np.zeros((n_groups, n_groups), dtype=int)
        for gp, gc in zip(groups_prev, groups_curr):
            if 0 <= gp < n_groups and 0 <= gc < n_groups:
                matrix[gp][gc] += 1

        print(f"\n  D{t-1} → D{t}  (n={len(common_users):,} users, {len(curr_users):,} total in D{t})")
        print(f"  Returning users: {len(common_users)/len(curr_users)*100:.1f}%")

        # Print matrix with percentages (row-normalized)
        header = "  prev\\curr " + "".join(f"  g{c:d}   " for c in range(n_groups))
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in range(n_groups):
            row_sum = matrix[r].sum()
            if row_sum == 0:
                pct = ["  0.0%" ] * n_groups
            else:
                pct = [f"{matrix[r][c]/row_sum*100:5.1f}%" for c in range(n_groups)]
            print(f"  g{r} (prev)  " + "  ".join(pct) + f"  (n={row_sum:,})")

        diag_pct = np.diag(matrix).sum() / matrix.sum() * 100
        print(f"  → Overall diagonal (stable group): {diag_pct:.1f}%")


# ─────────── Module B: history length at test time per group ────────────────

def module_B_test_history_distribution(data_root, dataset, periods):
    """
    在 test group 文件中，计算每个group的实际 history_len 分布。

    核心问题：测试时"cold group"是否真的有很短的history？
    注意：history最多10条，且跨period，所以即使是D{t}的group4用户，
    其history也可能来自D{t-1}的交互。
    """
    print_section("Module B: History Length Distribution at Test Time (per group)")
    print("  history_len = len(history_item_id) in each test row, max=10")
    print("  group 1=most active (test file numbering), group 5=least active")

    for t in periods:
        groups_dir = os.path.join(data_root, f"D{t}", "groups")
        group_files = sorted(glob.glob(os.path.join(groups_dir, "*.csv")))
        if not group_files:
            continue

        print(f"\n  D{t} test groups:")
        print(f"  {'Group file':<32} {'n_rows':>7} {'hist_med':>9} {'hist_mean':>10} {'hist_p10':>9} {'hist_p90':>9} {'hist=0%':>8} {'hist=10%':>9}")
        print(f"  {'-'*88}")

        for gf in group_files:
            gdf = pd.read_csv(gf)
            gdf["history_len"] = gdf["history_item_id"].apply(count_history_length)
            hl = gdf["history_len"]
            fname = os.path.basename(gf)
            zero_pct = (hl == 0).mean() * 100
            full_pct = (hl == 10).mean() * 100
            print(
                f"  {fname:<32} {len(gdf):>7,} {hl.median():>9.1f} {hl.mean():>10.2f} "
                f"{hl.quantile(0.1):>9.1f} {hl.quantile(0.9):>9.1f} {zero_pct:>7.1f}% {full_pct:>8.1f}%"
            )


# ─────────── Module C: new vs returning users per group ─────────────────────

def module_C_new_vs_returning(data_root, dataset, periods):
    """
    对 D{t} 每个 group，统计该 group 中有多少用户是全新的
    （即在 D0..D{t-1} 中从未出现过）。

    核心问题：各group中真正"冷启动"用户的占比。
    - 若 group4 中 new user 占比高 → cold group 确实是新用户 → forced routing 有语义
    - 若各group new user 占比相近 → group标签与新用户无关 → forced routing无意义
    """
    print_section("Module C: New vs Returning Users per Group per Period")
    print("  'New' = user_id never appeared in any prior period D0..D{t-1}")
    print("  group idx from user_group_map.json: 0=most active, 4=least active")

    seen_users: set = set()

    for t in periods:
        df = load_period_df(data_root, t, dataset)
        if df is None:
            if t == 0:
                seen_users = set()
            continue

        gmap = load_group_map(os.path.join(data_root, f"D{t}"))
        if gmap is None:
            if t == 0:
                seen_users.update(df["user_id"].astype(str).unique())
            continue

        df["user_str"] = df["user_id"].astype(str)
        df["group_idx"] = df["user_str"].map(gmap)

        curr_users = df[["user_str", "group_idx"]].drop_duplicates("user_str")
        curr_users["is_new"] = ~curr_users["user_str"].isin(seen_users)

        print(f"\n  D{t}  (total unique users: {len(curr_users):,})")
        print(f"  {'Group':<8} {'Users':>8} {'New':>8} {'New%':>8} {'Returning':>10}")
        print(f"  {'-'*46}")

        if curr_users["group_idx"].notna().any():
            for g, gdf in curr_users.groupby("group_idx"):
                n_new = gdf["is_new"].sum()
                print(
                    f"  {int(g):<8} {len(gdf):>8,} {n_new:>8,} {n_new/len(gdf)*100:>7.1f}% {len(gdf)-n_new:>10,}"
                )
        else:
            n_new = curr_users["is_new"].sum()
            print(f"  (no group map)  new={n_new:,} / {len(curr_users):,} ({n_new/len(curr_users)*100:.1f}%)")

        # Update seen set
        seen_users.update(curr_users["user_str"].tolist())


# ─────────── Module D: item overlap across groups (Jaccard) ─────────────────

def module_D_item_overlap(data_root, dataset, periods):
    """
    对 D{t} 每个 group，收集该 group 所有交互过的 item_id 集合，
    计算两两 group 之间的 Jaccard 相似度。

    核心问题：各 group 的 item 消费行为是否有显著差异？
    - 高 Jaccard → group 间 item 偏好高度重叠 → 很难为不同 group 专门化 memory keys
    - 低 Jaccard → group 间行为差异大 → forced routing 有潜力

    同时输出每个 group 的 item 集大小 和 独占 item 占比。
    """
    print_section("Module D: Item Overlap Across Groups (Jaccard Similarity)")
    print("  item_id = target item per interaction row")
    print("  group idx: 0=most active, 4=least active")

    for t in periods:
        df = load_period_df(data_root, t, dataset)
        if df is None:
            continue
        gmap = load_group_map(os.path.join(data_root, f"D{t}"))
        if gmap is None:
            continue

        df["group_idx"] = df["user_id"].astype(str).map(gmap)
        df_valid = df.dropna(subset=["group_idx"])

        groups = sorted(df_valid["group_idx"].unique())
        group_items = {}
        for g in groups:
            gdf = df_valid[df_valid["group_idx"] == g]
            group_items[g] = set(gdf["item_id"].astype(str).tolist())

        print(f"\n  D{t}")
        # Item set sizes and exclusive items
        all_items = set.union(*group_items.values()) if group_items else set()
        print(f"  Total unique items across all groups: {len(all_items):,}")
        print(f"  {'Group':<8} {'Items':>8} {'Exclusive':>10} {'Exclusive%':>11}")
        print(f"  {'-'*40}")
        for g in groups:
            others = set.union(*[s for gg, s in group_items.items() if gg != g]) if len(group_items) > 1 else set()
            exclusive = group_items[g] - others
            print(
                f"  {int(g):<8} {len(group_items[g]):>8,} {len(exclusive):>10,} {len(exclusive)/max(1,len(group_items[g]))*100:>10.1f}%"
            )

        # Jaccard matrix
        n = len(groups)
        print(f"\n  Jaccard similarity matrix (row×col = group pair):")
        header = "  g\\g  " + "".join(f"  g{int(c):<4}" for c in groups)
        print(header)
        for g1 in groups:
            row_str = f"  g{int(g1):<4} "
            for g2 in groups:
                if g1 == g2:
                    row_str += "  1.000"
                else:
                    inter = len(group_items[g1] & group_items[g2])
                    union = len(group_items[g1] | group_items[g2])
                    j = inter / union if union > 0 else 0.0
                    row_str += f"  {j:.3f}"
            print(row_str)


# ─────────────────── Module E: prev_interactions distribution shape ──────────

def module_E_prev_interactions_distribution(data_root, dataset, periods):
    """
    对 D{t}（t>=1）中每个 group，输出 prev_interactions 的分布分位点。

    核心问题：各 group 之间的 prev_interactions 是否有明显分离？
    - 若各group分位点差异很小（power-law导致大家都在1-3区间）→ group划分语义弱
    - 若各group分位点有明显区分 → group划分有意义

    同时输出：在 prev_interactions=0 的用户中（真正新用户），各group占比
    """
    print_section("Module E: Prev-Interactions Distribution Shape per Group")
    print("  Shows whether group boundaries correspond to meaningful behavior gaps")

    for t in periods:
        if t == 0:
            continue
        prev_csv = os.path.join(data_root, f"D{t-1}", f"{dataset}.csv")
        df = load_period_df(data_root, t, dataset)
        if df is None or not os.path.isfile(prev_csv):
            continue
        gmap = load_group_map(os.path.join(data_root, f"D{t}"))
        if gmap is None:
            continue

        prev_df = pd.read_csv(prev_csv)
        prev_df["history_len"] = prev_df["history_item_id"].apply(count_history_length)
        user_pop = prev_df.groupby("user_id")["history_len"].max().rename("prev_interactions").reset_index()

        curr_users = pd.DataFrame({"user_id": df["user_id"].unique()})
        user_df = curr_users.merge(user_pop, on="user_id", how="left")
        user_df["prev_interactions"] = user_df["prev_interactions"].fillna(0)
        user_df["group_idx"] = user_df["user_id"].astype(str).map(gmap)

        print(f"\n  D{t}  prev_interactions from D{t-1}")
        print(f"  {'Group':<8} {'n':>7} {'p0':>5} {'p10':>5} {'p25':>5} {'p50':>6} {'p75':>6} {'p90':>6} {'p99':>6} {'max':>6}")
        print(f"  {'-'*60}")

        for g, gdf in user_df.groupby("group_idx"):
            pi = gdf["prev_interactions"]
            print(
                f"  {int(g):<8} {len(gdf):>7,} "
                f"{pi.quantile(0):>5.0f} {pi.quantile(0.1):>5.0f} {pi.quantile(0.25):>5.0f} "
                f"{pi.quantile(0.5):>6.0f} {pi.quantile(0.75):>6.0f} {pi.quantile(0.9):>6.0f} "
                f"{pi.quantile(0.99):>6.0f} {pi.max():>6.0f}"
            )

        # Boundary analysis: overlap between adjacent groups
        print(f"\n  Adjacent group boundary overlap (D{t}):")
        group_ranges = {}
        for g, gdf in user_df.groupby("group_idx"):
            pi = gdf["prev_interactions"]
            group_ranges[int(g)] = (pi.min(), pi.max(), pi.median())

        sorted_groups = sorted(group_ranges.keys())
        for i in range(len(sorted_groups) - 1):
            g1, g2 = sorted_groups[i], sorted_groups[i+1]
            min1, max1, med1 = group_ranges[g1]
            min2, max2, med2 = group_ranges[g2]
            overlap_start = max(min1, min2)
            overlap_end = min(max1, max2)
            has_overlap = overlap_start <= overlap_end
            print(
                f"  g{g1}[{min1:.0f}-{max1:.0f}] vs g{g2}[{min2:.0f}-{max2:.0f}]: "
                f"{'OVERLAP [{:.0f}-{:.0f}]'.format(overlap_start, overlap_end) if has_overlap else 'no overlap'}"
            )


# ──────── Module F: target item popularity per group (长尾分析) ─────────────

def module_F_target_item_popularity(data_root, dataset, periods):
    """
    核心问题：group 0（最活跃用户）的 next item 是否更多是冷门/长尾 item？

    方法：
    1. 用全量数据（D0-D4合并）计算每个 item_id 的全局交互频次，得到 popularity rank
       - pop_pct: 0.0=最热门(头部), 1.0=最冷门(长尾)
       - 长尾定义：pop_pct > 0.8（即全局频次排名后20%的item）
    2. 对每个 period 和 group，分析 target item（item_id列）的 popularity 分布
    3. 额外输出：各 group 的 target item 中 head/mid/tail 的占比

    结论解读：
    - 若 group0 的 tail_pct 显著高于其他 group → warm user 确实更多购买长尾item
      → recall@20 低是任务难度导致，memory layer 应该向 item-side routing 设计
    - 若各 group 的 tail_pct 相近 → 长尾不是主要原因，需要其他解释
    """
    print_section("Module F: Target Item Popularity per Group (长尾分析)")
    print("  Global popularity computed across ALL periods (D0-D4)")
    print("  pop_pct: 0.0=head(most popular), 1.0=tail(least popular)")
    print("  Tail = pop_pct > 0.8 | Mid = 0.4~0.8 | Head = pop_pct <= 0.4")
    print("  group idx: 0=most active user, 4=least active user")

    # ── Step 1: compute global item popularity across all periods ──
    all_dfs = []
    for t in periods:
        df = load_period_df(data_root, t, dataset)
        if df is not None:
            all_dfs.append(df[["item_id"]])

    if not all_dfs:
        print("  [SKIP] No data found")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    item_freq = combined["item_id"].astype(str).value_counts().rename("global_freq")
    n_items = len(item_freq)

    # pop_rank: 1 = most popular; pop_pct: 0.0=head, 1.0=tail
    item_pop_df = item_freq.reset_index()
    item_pop_df.columns = ["item_id", "global_freq"]
    item_pop_df = item_pop_df.sort_values("global_freq", ascending=False).reset_index(drop=True)
    item_pop_df["pop_rank"] = item_pop_df.index + 1
    item_pop_df["pop_pct"] = (item_pop_df["pop_rank"] - 1) / max(n_items - 1, 1)

    pop_map = dict(zip(item_pop_df["item_id"], item_pop_df["pop_pct"]))
    freq_map = dict(zip(item_pop_df["item_id"], item_pop_df["global_freq"]))

    print(f"\n  Global item catalog: {n_items:,} unique items")
    print(f"  Freq range: min={item_pop_df['global_freq'].min()}, "
          f"median={item_pop_df['global_freq'].median():.0f}, "
          f"max={item_pop_df['global_freq'].max()}")

    # ── Step 2: per-period, per-group analysis ──
    for t in periods:
        df = load_period_df(data_root, t, dataset)
        if df is None:
            continue
        gmap = load_group_map(os.path.join(data_root, f"D{t}"))
        if gmap is None:
            print(f"\n  D{t}: [SKIP] no user_group_map.json")
            continue

        df["group_idx"] = df["user_id"].astype(str).map(gmap)
        df["item_str"] = df["item_id"].astype(str)
        df["pop_pct"] = df["item_str"].map(pop_map)
        df["global_freq"] = df["item_str"].map(freq_map)
        df_valid = df.dropna(subset=["group_idx", "pop_pct"])

        print(f"\n  D{t}  (rows with valid group+popularity: {len(df_valid):,}/{len(df):,})")
        print(f"  {'Group':<8} {'n_rows':>8} {'pop_med':>8} {'pop_mean':>9} "
              f"{'head%':>7} {'mid%':>7} {'tail%':>7} {'freq_med':>9}")
        print(f"  {'-'*67}")

        for g, gdf in df_valid.groupby("group_idx"):
            pp = gdf["pop_pct"]
            freq = gdf["global_freq"]
            head_pct = (pp <= 0.4).mean() * 100
            mid_pct  = ((pp > 0.4) & (pp <= 0.8)).mean() * 100
            tail_pct = (pp > 0.8).mean() * 100
            print(
                f"  {int(g):<8} {len(gdf):>8,} {pp.median():>8.3f} {pp.mean():>9.3f} "
                f"{head_pct:>6.1f}% {mid_pct:>6.1f}% {tail_pct:>6.1f}% {freq.median():>9.1f}"
            )

        # Overall for this period
        pp_all = df_valid["pop_pct"]
        freq_all = df_valid["global_freq"]
        print(
            f"  {'ALL':<8} {len(df_valid):>8,} {pp_all.median():>8.3f} {pp_all.mean():>9.3f} "
            f"{(pp_all<=0.4).mean()*100:>6.1f}% "
            f"{((pp_all>0.4)&(pp_all<=0.8)).mean()*100:>6.1f}% "
            f"{(pp_all>0.8).mean()*100:>6.1f}% {freq_all.median():>9.1f}"
        )

    # ── Step 3: also check test group files ──
    print(f"\n  ── Test group files (using same global popularity map) ──")
    for t in periods:
        groups_dir = os.path.join(data_root, f"D{t}", "groups")
        group_files = sorted(glob.glob(os.path.join(groups_dir, "*.csv")))
        if not group_files:
            continue

        print(f"\n  D{t} test groups:")
        print(f"  {'File':<32} {'n_rows':>7} {'pop_med':>8} {'head%':>7} {'mid%':>7} {'tail%':>7} {'freq_med':>9}")
        print(f"  {'-'*75}")

        for gf in group_files:
            gdf = pd.read_csv(gf)
            gdf["item_str"] = gdf["item_id"].astype(str)
            gdf["pop_pct"] = gdf["item_str"].map(pop_map)
            gdf["global_freq"] = gdf["item_str"].map(freq_map)
            gdf_v = gdf.dropna(subset=["pop_pct"])
            if len(gdf_v) == 0:
                continue
            pp = gdf_v["pop_pct"]
            freq = gdf_v["global_freq"]
            fname = os.path.basename(gf)
            print(
                f"  {fname:<32} {len(gdf_v):>7,} {pp.median():>8.3f} "
                f"{(pp<=0.4).mean()*100:>6.1f}% "
                f"{((pp>0.4)&(pp<=0.8)).mean()*100:>6.1f}% "
                f"{(pp>0.8).mean()*100:>6.1f}% {freq.median():>9.1f}"
            )


# ──────────────────────────────── main ──────────────────────────────────────

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
    ap.add_argument("--modules", default="orig,A,B,C,D,E,F",
                    help="Comma-separated modules to run: orig,A,B,C,D,E,F")
    args = ap.parse_args()

    periods = [int(x) for x in args.periods.split(",")]
    modules = [m.strip() for m in args.modules.split(",")]

    print(f"Data root : {args.data_root}")
    print(f"Dataset   : {args.dataset}")
    print(f"Periods   : D{periods[0]} .. D{periods[-1]}")
    print(f"Modules   : {modules}")

    if "orig" in modules:
        for t in periods:
            stats_for_period_original(t, args.data_root, args.dataset)

    if "A" in modules:
        module_A_group_transition(args.data_root, args.dataset, periods)

    if "B" in modules:
        module_B_test_history_distribution(args.data_root, args.dataset, periods)

    if "C" in modules:
        module_C_new_vs_returning(args.data_root, args.dataset, periods)

    if "D" in modules:
        module_D_item_overlap(args.data_root, args.dataset, periods)

    if "E" in modules:
        module_E_prev_interactions_distribution(args.data_root, args.dataset, periods)

    if "F" in modules:
        module_F_target_item_popularity(args.data_root, args.dataset, periods)

    print(f"\n{'='*70}")
    print("  Done")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
