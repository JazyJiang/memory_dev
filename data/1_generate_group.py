import pandas as pd
import ast
import os


def count_history_length(history_str):
    """Safely parse history field and return length"""
    try:
        items = ast.literal_eval(history_str)
        return len(items) if isinstance(items, list) else 0
    except Exception:
        return 0


def group_test_by_prev_period(
    prev_period_path,
    curr_period_path,
    k=5,
    output_dir=None,
):
    """
    For period D_t:
    - use D_{t-1} to compute user popularity
    - derive k user groups
    - apply grouping to interactions in D_t
    """

    # ===== load data =====
    prev_df = pd.read_csv(prev_period_path)
    curr_df = pd.read_csv(curr_period_path)

    # ===== 1️⃣ user popularity from D_{t-1} =====
    prev_df["history_len"] = prev_df["history_item_id"].apply(count_history_length)

    user_pop = (
        prev_df.groupby("user_id")["history_len"]
        .max()
        .rename("prev_interactions")
        .reset_index()
    )

    # ===== 2️⃣ users appearing in D_t =====
    curr_users = pd.DataFrame({"user_id": curr_df["user_id"].unique()})
    user_df = curr_users.merge(user_pop, on="user_id", how="left")
    user_df["prev_interactions"] = user_df["prev_interactions"].fillna(0)

    # ===== 3️⃣ quantile grouping by popularity =====
    user_df = user_df.sort_values(
        "prev_interactions", ascending=False
    ).reset_index(drop=True)

    user_df["user_group"] = (
        pd.qcut(user_df.index, q=k, labels=False, duplicates="drop") + 1
    )

    # ===== 4️⃣ apply grouping to D_t interactions =====
    curr_df = curr_df.merge(
        user_df[["user_id", "user_group"]],
        on="user_id",
        how="left",
    )

    # ===== 5️⃣ write grouped test files =====
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(curr_period_path), "groups"
        )
    os.makedirs(output_dir, exist_ok=True)

    for g in sorted(curr_df["user_group"].dropna().unique()):
        subset = curr_df[curr_df["user_group"] == g].drop(columns=["user_group"])
        out_path = os.path.join(output_dir, f"test_user_group{int(g)}.csv")
        subset.to_csv(out_path, index=False)
        print(f"✅ D_t group {g}: {len(subset)} rows → {out_path}")

    return user_df[["user_id", "user_group"]]


# DATA_ROOT = "/home/xinyulin/context/data/2026-01-25_5period"
DATA_ROOT ="/mlx_devbox/users/zhuosong.jiang/playground/memory_dev/data"
for t in range(1, 5):
    prev_path = f"{DATA_ROOT}/D{t-1}/Toys_and_Games_5_2016-10-2018-11.csv"
    curr_path = f"{DATA_ROOT}/D{t}/Toys_and_Games_5_2016-10-2018-11.csv"

    print(f"\n=== Grouping D{t} using D{t-1} ===")
    group_test_by_prev_period(
        prev_period_path=prev_path,
        curr_period_path=curr_path,
        k=5,
    )