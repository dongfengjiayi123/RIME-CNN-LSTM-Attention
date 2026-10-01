"""
生成结果对比图和模型架构图
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RESULT_DIR = config.RESULT_DIR
ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(ASSET_DIR, exist_ok=True)


def plot_baseline_comparison():
    """绘制 baseline 对比柱状图（使用论文中的真实数据）"""
    models = ["XGBoost", "BERT4Rec", "MMoE", "LightGCN", "RIME-Std", "Ours"]
    # 三数据集平均 AUC（来自论文 Table 1）
    avg_auc = [0.777, 0.830, 0.809, 0.802, 0.838, 0.863]
    avg_f1 = [0.733, 0.798, 0.782, 0.773, 0.814, 0.842]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#95a5a6", "#95a5a6", "#95a5a6", "#95a5a6", "#95a5a6", "#e74c3c"]

    # AUC 对比
    bars1 = axes[0].bar(models, avg_auc, color=colors, edgecolor="white", linewidth=1.2)
    axes[0].set_ylabel("AUC-ROC", fontsize=13, fontweight="bold")
    axes[0].set_title("Average AUC-ROC across 3 datasets", fontsize=14, fontweight="bold")
    axes[0].set_ylim(0.70, 0.90)
    axes[0].tick_params(axis="x", rotation=20)
    for bar, val in zip(bars1, avg_auc):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # F1 对比
    bars2 = axes[1].bar(models, avg_f1, color=colors, edgecolor="white", linewidth=1.2)
    axes[1].set_ylabel("F1-Score", fontsize=13, fontweight="bold")
    axes[1].set_title("Average F1-Score across 3 datasets", fontsize=14, fontweight="bold")
    axes[1].set_ylim(0.68, 0.88)
    axes[1].tick_params(axis="x", rotation=20)
    for bar, val in zip(bars2, avg_f1):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 图例
    ours_patch = mpatches.Patch(color="#e74c3c", label="Ours (RIME-CNN-LSTM-Attention)")
    baseline_patch = mpatches.Patch(color="#95a5a6", label="Baselines")
    fig.legend(handles=[ours_patch, baseline_patch], loc="upper center",
               ncol=2, fontsize=11, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    path = os.path.join(ASSET_DIR, "baseline_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"对比图已保存: {path}")


def plot_per_dataset():
    """绘制每个数据集上的详细对比"""
    datasets = ["Tmall", "Taobao", "CIKM19"]
    ours_auc = [0.863, 0.844, 0.879]
    bert4rec_auc = [0.833, 0.803, 0.852]
    xgboost_auc = [0.779, 0.752, 0.799]

    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars1 = ax.bar(x - width, xgboost_auc, width, label="XGBoost", color="#95a5a6")
    bars2 = ax.bar(x, bert4rec_auc, width, label="BERT4Rec", color="#3498db")
    bars3 = ax.bar(x + width, ours_auc, width, label="Ours", color="#e74c3c")

    ax.set_ylabel("AUC-ROC", fontsize=13, fontweight="bold")
    ax.set_title("AUC-ROC Comparison on Each Dataset", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylim(0.70, 0.92)
    ax.legend(fontsize=11)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(ASSET_DIR, "per_dataset_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"分数据集对比图已保存: {path}")


def plot_architecture():
    """绘制模型架构图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def draw_box(x, y, w, h, text, color="#3498db", text_color="white", fontsize=10):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold")

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    # 标题
    ax.text(7, 9.5, "RIME-CNN-LSTM-Attention Architecture", ha="center",
            fontsize=16, fontweight="bold", color="#2c3e50")

    # 输入层
    draw_box(0.5, 7.5, 2.5, 1.0, "Behavior\nSequence", "#2ecc71", fontsize=9)
    draw_box(3.5, 7.5, 2.5, 1.0, "Item\nSequence", "#2ecc71", fontsize=9)
    draw_box(6.5, 7.5, 2.5, 1.0, "Time\nInterval", "#2ecc71", fontsize=9)

    # Embedding 层
    draw_box(0.5, 5.8, 2.5, 1.0, "Behavior\nEmbedding", "#f39c12", fontsize=9)
    draw_box(3.5, 5.8, 2.5, 1.0, "Item\nEmbedding", "#f39c12", fontsize=9)
    draw_box(6.5, 5.8, 2.5, 1.0, "Time\nEmbedding", "#f39c12", fontsize=9)

    # 拼接
    draw_box(3.5, 4.2, 5.5, 0.9, "Concatenate (Multi-view Fusion)", "#e67e22", fontsize=10)

    # CNN
    draw_box(3.5, 2.8, 5.5, 0.9, "CNN (Local Feature Extraction)", "#9b59b6", fontsize=10)

    # LSTM
    draw_box(3.5, 1.4, 5.5, 0.9, "LSTM (Temporal Dependency)", "#3498db", fontsize=10)

    # Attention
    draw_box(3.5, 0.0, 5.5, 0.9, "Attention (Key Feature Weighting)", "#e74c3c", fontsize=10)

    # 输出
    draw_box(10.5, 0.0, 2.5, 0.9, "FC + Sigmoid\n(Purchase Prob.)", "#1abc9c", fontsize=9)

    # 箭头
    for x in [1.75, 4.75, 7.75]:
        draw_arrow(x, 7.5, x, 6.8)
    draw_arrow(4.75, 5.8, 5.0, 5.1)
    draw_arrow(6.25, 5.8, 6.25, 5.1)
    draw_arrow(6.25, 4.2, 6.25, 3.7)
    draw_arrow(6.25, 2.8, 6.25, 2.3)
    draw_arrow(6.25, 1.4, 6.25, 0.9)
    draw_arrow(9.0, 0.45, 10.5, 0.45)

    plt.tight_layout()
    path = os.path.join(ASSET_DIR, "model_architecture.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"架构图已保存: {path}")


if __name__ == "__main__":
    plot_baseline_comparison()
    plot_per_dataset()
    plot_architecture()
    print("所有图表生成完成！")
