"""
主运行脚本：数据加载 → 训练 → 评估 → 结果保存
"""
import os
import sys
import json
import argparse
import pandas as pd

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.dataset import preprocess_raw_data, build_dataloader, train_test_split
from src.train import Trainer
from src.evaluate import compute_metrics, compare_baselines


def load_and_preprocess():
    """加载并预处理数据"""
    print(f"加载数据集: {config.DATASET}")
    if not os.path.exists(config.DATA_PATH):
        print(f"数据文件不存在: {config.DATA_PATH}")
        print("请先下载数据集并放到 data/ 目录下")
        print("Taobao User Behavior Dataset: https://tianchi.aliyun.com/dataset/649")
        sys.exit(1)

    df = pd.read_csv(config.DATA_PATH)
    print(f"原始数据行数: {len(df):,}")

    sequences, labels = preprocess_raw_data(
        df, min_user_behaviors=config.MIN_USER_BEHAVIORS
    )
    print(f"有效用户数: {len(labels):,}")
    print(f"正样本比例: {sum(labels)/len(labels):.2%}")

    return sequences, labels


def run_experiment():
    """运行完整实验"""
    # 加载数据
    sequences, labels = load_and_preprocess()

    # 划分训练集/测试集
    train_seq, train_label, test_seq, test_label = train_test_split(
        sequences, labels, test_ratio=0.2, seed=config.SEED
    )
    # 从训练集划分验证集
    val_split = int(len(train_label) * 0.1)
    val_seq, val_label = train_seq[:val_split], train_label[:val_split]
    train_seq, train_label = train_seq[val_split:], train_label[val_split:]

    print(f"训练集: {len(train_label)}, 验证集: {len(val_label)}, 测试集: {len(test_label)}")

    # 构建 DataLoader
    train_loader = build_dataloader(
        train_seq, train_label, config.BATCH_SIZE, config.MAX_SEQ_LEN, shuffle=True
    )
    val_loader = build_dataloader(
        val_seq, val_label, config.BATCH_SIZE, config.MAX_SEQ_LEN, shuffle=False
    )
    test_loader = build_dataloader(
        test_seq, test_label, config.BATCH_SIZE, config.MAX_SEQ_LEN, shuffle=False
    )

    # 训练
    trainer = Trainer(config, train_loader, val_loader)
    trainer.train()

    # 测试集评估
    trainer.load_model("best_model.pth")
    trainer.model.eval()

    import torch
    import numpy as np

    all_preds = []
    all_labels = []
    device = trainer.device
    with torch.no_grad():
        for batch in test_loader:
            behavior_seq = batch["behavior_seq"].to(device)
            item_seq = batch["item_seq"].to(device)
            time_seq = batch["time_seq"].to(device)
            mask = batch["mask"].to(device)
            labels_batch = batch["label"].squeeze(-1).to(device)

            outputs = trainer.model(behavior_seq, item_seq, time_seq, mask)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    test_metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    print("\n========== 测试集结果 ==========")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 保存结果
    result_path = os.path.join(config.RESULT_DIR, f"{config.DATASET}_results.json")
    with open(result_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n结果已保存: {result_path}")

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="taobao", help="数据集名称")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    args = parser.parse_args()

    config.DATASET = args.dataset
    config.DATA_PATH = os.path.join(config.DATA_DIR, f"{args.dataset}.csv")
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size

    run_experiment()
