"""
生成模拟电商用户行为数据集（用于演示和测试）
格式与 Taobao User Behavior Dataset 一致
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def generate_synthetic_data(num_users=5000, num_items=2000, avg_behaviors=30, seed=42):
    """生成模拟用户行为数据

    模拟逻辑：
    - 每个用户有若干行为序列
    - 行为类型：pv(点击)、fav(收藏)、cart(加购)、buy(购买)
    - 有购买行为的用户比例约 30%
    - 购买行为通常出现在序列末尾
    """
    np.random.seed(seed)

    records = []
    base_time = 1511539200  # 2017-11-25 00:00:00 时间戳

    for user_id in range(1, num_users + 1):
        # 该用户是否会购买
        will_buy = np.random.random() < 0.3
        # 行为次数
        n_behaviors = np.random.randint(5, avg_behaviors * 2)

        timestamps = sorted(
            base_time + np.random.randint(0, 86400 * 7, n_behaviors)
        )

        for i in range(n_behaviors):
            item_id = np.random.randint(1, num_items + 1)

            if will_buy and i == n_behaviors - 1:
                # 最后一个行为是购买
                behavior = "buy"
            elif will_buy and i == n_behaviors - 2:
                # 倒数第二个行为可能是加购
                behavior = np.random.choice(["cart", "fav", "pv"], p=[0.5, 0.3, 0.2])
            else:
                # 普通行为
                behavior = np.random.choice(
                    ["pv", "fav", "cart"], p=[0.80, 0.12, 0.08]
                )

            records.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "behavior_type": behavior,
                    "timestamp": timestamps[i],
                }
            )

    df = pd.DataFrame(records)
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("生成模拟数据集...")
    df = generate_synthetic_data(num_users=5000, num_items=2000)
    print(f"总行为记录数: {len(df):,}")
    print(f"用户数: {df['user_id'].nunique():,}")
    print(f"物品数: {df['item_id'].nunique():,}")
    print(f"行为分布:\n{df['behavior_type'].value_counts()}")

    # 保存为三个数据集（用同一份模拟数据，实际使用时替换为真实数据）
    for name in ["taobao", "tmall", "cikm19"]:
        path = os.path.join(config.DATA_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"已保存: {path}")

    print("模拟数据生成完成！")
