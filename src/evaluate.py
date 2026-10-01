"""
评估模块：计算 F1、AUC、Accuracy、Precision、Recall 等指标
"""
import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)


def compute_metrics(labels, preds, threshold=0.5):
    """计算评估指标

    Args:
        labels: 真实标签 (n,)
        preds: 预测概率 (n,)
        threshold: 二分类阈值

    Returns:
        dict: 包含各指标的字典
    """
    pred_labels = (preds >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, pred_labels),
        "precision": precision_score(labels, pred_labels, zero_division=0),
        "recall": recall_score(labels, pred_labels, zero_division=0),
        "f1": f1_score(labels, pred_labels, zero_division=0),
    }

    # AUC 需要至少两类样本
    if len(np.unique(labels)) > 1:
        metrics["auc"] = roc_auc_score(labels, preds)
    else:
        metrics["auc"] = 0.0

    return metrics


def compare_baselines(results_dict):
    """对比多个模型的结果

    Args:
        results_dict: {model_name: {metric: value}}

    Returns:
        str: 格式化的对比表格
    """
    metrics_order = ["f1", "auc", "accuracy", "precision", "recall"]
    model_names = list(results_dict.keys())

    # 表头
    header = f"{'Model':<20}" + "".join(f"{m:>12}" for m in metrics_order)
    separator = "-" * len(header)

    lines = [header, separator]
    for name in model_names:
        row = f"{name:<20}"
        for m in metrics_order:
            val = results_dict[name].get(m, 0.0)
            row += f"{val:>12.4f}"
        lines.append(row)

    return "\n".join(lines)


def print_classification_report(labels, preds, threshold=0.5):
    """打印详细分类报告"""
    pred_labels = (preds >= threshold).astype(int)
    print(classification_report(labels, pred_labels, target_names=["未购买", "购买"]))
