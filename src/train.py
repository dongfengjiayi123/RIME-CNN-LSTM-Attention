"""
训练模块
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .model import RIME_CNN_LSTM_Attention
from .evaluate import compute_metrics


class Trainer:
    """模型训练器"""

    def __init__(self, config, train_loader, val_loader=None):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(
            config.DEVICE if torch.cuda.is_available() else "cpu"
        )

        # 初始化模型
        self.model = RIME_CNN_LSTM_Attention(
            num_behavior_types=config.NUM_BEHAVIOR_TYPES,
            num_items=10000,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            cnn_kernel_size=config.CNN_KERNEL_SIZE,
            cnn_num_filters=config.CNN_NUM_FILTERS,
            attention_dim=config.ATTENTION_DIM,
            max_seq_len=config.MAX_SEQ_LEN,
        ).to(self.device)

        # 损失函数与优化器
        self.criterion = nn.BCELoss()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3
        )

        # 训练记录
        self.train_losses = []
        self.val_aucs = []
        self.best_auc = 0.0
        self.patience_counter = 0

    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            behavior_seq = batch["behavior_seq"].to(self.device)
            item_seq = batch["item_seq"].to(self.device)
            time_seq = batch["time_seq"].to(self.device)
            mask = batch["mask"].to(self.device)
            labels = batch["label"].squeeze(-1).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(behavior_seq, item_seq, time_seq, mask)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        return avg_loss, metrics

    def validate(self):
        """验证"""
        if self.val_loader is None:
            return None

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                behavior_seq = batch["behavior_seq"].to(self.device)
                item_seq = batch["item_seq"].to(self.device)
                time_seq = batch["time_seq"].to(self.device)
                mask = batch["mask"].to(self.device)
                labels = batch["label"].squeeze(-1).to(self.device)

                outputs = self.model(behavior_seq, item_seq, time_seq, mask)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        return metrics

    def train(self):
        """完整训练流程"""
        print(f"训练设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, self.config.EPOCHS + 1):
            train_loss, train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            val_metrics = self.validate()
            if val_metrics:
                self.val_aucs.append(val_metrics["auc"])
                self.scheduler.step(val_metrics["auc"])

                print(
                    f"Epoch {epoch}/{self.config.EPOCHS} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train AUC: {train_metrics['auc']:.4f} | "
                    f"Val AUC: {val_metrics['auc']:.4f} | "
                    f"Val F1: {val_metrics['f1']:.4f}"
                )

                # 保存最佳模型
                if val_metrics["auc"] > self.best_auc:
                    self.best_auc = val_metrics["auc"]
                    self.save_model("best_model.pth")
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.config.PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                print(
                    f"Epoch {epoch}/{self.config.EPOCHS} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train AUC: {train_metrics['auc']:.4f}"
                )

        print(f"\n训练完成，最佳验证 AUC: {self.best_auc:.4f}")
        return self.best_auc

    def save_model(self, filename):
        """保存模型"""
        path = os.path.join(self.config.MODEL_DIR, filename)
        # 只保存可序列化的配置项（排除 module、function 等对象）
        import types
        config_dict = {}
        for k, v in vars(self.config).items():
            if k.startswith("_"):
                continue
            if isinstance(v, (int, float, str, bool, list, dict, tuple)):
                config_dict[k] = v
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_auc": self.best_auc,
                "config": config_dict,
            },
            path,
        )
        print(f"模型已保存: {path}")

    def load_model(self, filename):
        """加载模型"""
        path = os.path.join(self.config.MODEL_DIR, filename)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_auc = checkpoint["best_auc"]
        print(f"模型已加载: {path}, AUC: {self.best_auc:.4f}")
