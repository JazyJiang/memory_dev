import pandas as pd
import ast
import os
import json
import numpy as np
import fire

def count_history_length(history_str):
    try:
        items = ast.literal_eval(history_str)
        return len(items) if isinstance(items, list) else 0
    except Exception:
        return 0

def generate_user_group_map(data_root="/workspace/jiangzhuosong/memory_dev/data", dataset_name="Toys_and_Games_5_2016-10-2018-11", n_groups=5):
    # Process D0 (use current period interaction count)
    d0_path = os.path.join(data_root, "D0", f"{dataset_name}.csv")
    if os.path.exists(d0_path):
        print("Processing D0 (Self-Grouping based on frequency)...")
        df = pd.read_csv(d0_path)
        
        # Calculate user frequency in D0
        user_freq = df['user_id'].value_counts().reset_index()
        user_freq.columns = ['user_id', 'freq']
        
        # Group users by frequency
        user_freq = user_freq.sort_values('freq', ascending=False).reset_index(drop=True)
        
        try:
            user_freq["group_idx"] = pd.qcut(user_freq.index, q=n_groups, labels=False, duplicates="drop")
        except ValueError:
            user_freq["group_idx"] = 0
            sz = len(user_freq) // n_groups
            for i in range(n_groups):
                st = i * sz
                ed = (i + 1) * sz if i < n_groups - 1 else len(user_freq)
                user_freq.iloc[st:ed, user_freq.columns.get_loc("group_idx")] = i

        group_map = dict(zip(user_freq["user_id"].astype(str), user_freq["group_idx"]))
        
        out_path = os.path.join(data_root, "D0", "user_group_map.json")
        with open(out_path, 'w') as f:
            json.dump(group_map, f)
        print(f"Saved D0 map to {out_path}")

    # Process D1 to D4
    for t in range(1, 5):
        prev_path = os.path.join(data_root, f"D{t-1}", f"{dataset_name}.csv")
        curr_path = os.path.join(data_root, f"D{t}", f"{dataset_name}.csv")
        
        if not os.path.exists(prev_path) or not os.path.exists(curr_path):
            continue
            
        print(f"Processing D{t} using D{t-1}...")
        
        prev_df = pd.read_csv(prev_path)
        curr_df = pd.read_csv(curr_path)
        
        # Calculate popularity from prev period
        prev_df["history_len"] = prev_df["history_item_id"].apply(count_history_length)
        user_pop = (
            prev_df.groupby("user_id")["history_len"]
            .max()
            .rename("prev_interactions")
            .reset_index()
        )
        
        # Merge with current users
        curr_users = pd.DataFrame({"user_id": curr_df["user_id"].unique()})
        user_df = curr_users.merge(user_pop, on="user_id", how="left")
        user_df["prev_interactions"] = user_df["prev_interactions"].fillna(0)
        
        # Grouping
        user_df = user_df.sort_values("prev_interactions", ascending=False).reset_index(drop=True)
        user_df["group_idx"] = pd.qcut(user_df.index, q=n_groups, labels=False, duplicates="drop")
        
        group_map = dict(zip(user_df["user_id"].astype(str), user_df["group_idx"]))
        
        out_path = os.path.join(data_root, f"D{t}", "user_group_map.json")
        with open(out_path, 'w') as f:
            json.dump(group_map, f)
        print(f"Saved D{t} map to {out_path}")

if __name__ == "__main__":
    fire.Fire(generate_user_group_map)
