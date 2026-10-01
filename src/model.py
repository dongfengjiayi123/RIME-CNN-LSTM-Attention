"""
RIME-CNN-LSTM-Attention 模型定义

架构：
1. Embedding 层：行为类型 + 物品 + 时间间隔
2. CNN 层：提取行为序列局部特征
3. LSTM 层：建模时序依赖
4. Attention 层：加权关键行为特征
5. 全连接层：输出购买概率
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RIME_CNN_LSTM_Attention(nn.Module):
    """RIME: Recommendation via Integration of Multi-view Encoding"""

    def __init__(
        self,
        num_behavior_types=4,
        num_items=10000,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        cnn_kernel_size=3,
        cnn_num_filters=64,
        attention_dim=64,
        max_seq_len=50,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # ========== Embedding 层 ==========
        self.behavior_embedding = nn.Embedding(num_behavior_types, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.time_embedding = nn.Linear(1, embedding_dim)

        # 融合后的输入维度
        input_dim = embedding_dim * 3

        # ========== CNN 层：提取局部特征 ==========
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=cnn_num_filters,
                kernel_size=cnn_kernel_size,
                padding=cnn_kernel_size // 2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
        )
        self.cnn_out_dim = cnn_num_filters

        # ========== LSTM 层：建模时序依赖 ==========
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # ========== Attention 层 ==========
        self.attention_W = nn.Linear(hidden_dim, attention_dim)
        self.attention_v = nn.Linear(attention_dim, 1, bias=False)

        # ========== 输出层 ==========
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, behavior_seq, item_seq, time_seq, mask=None):
        """
        Args:
            behavior_seq: (batch, seq_len) 行为类型序列
            item_seq: (batch, seq_len) 物品序列
            time_seq: (batch, seq_len) 时间间隔序列
            mask: (batch, seq_len) 有效位置掩码
        """
        batch_size = behavior_seq.size(0)

        # ========== Embedding ==========
        behavior_emb = self.behavior_embedding(behavior_seq)  # (B, L, D)
        item_emb = self.item_embedding(item_seq)  # (B, L, D)
        time_emb = self.time_embedding(time_seq.unsqueeze(-1))  # (B, L, D)

        # 拼接多视图嵌入
        x = torch.cat([behavior_emb, item_emb, time_emb], dim=-1)  # (B, L, 3D)

        # ========== CNN 局部特征提取 ==========
        x = x.permute(0, 2, 1)  # (B, 3D, L) for Conv1d
        cnn_out = self.cnn(x)  # (B, cnn_filters, L')
        cnn_out = cnn_out.permute(0, 2, 1)  # (B, L', cnn_filters)

        # ========== LSTM 时序建模 ==========
        lstm_out, (h_n, c_n) = self.lstm(cnn_out)  # (B, L', hidden)

        # ========== Attention 加权 ==========
        # 计算注意力权重
        energy = torch.tanh(self.attention_W(lstm_out))  # (B, L', att_dim)
        attention_scores = self.attention_v(energy).squeeze(-1)  # (B, L')

        # 应用 mask（如果有）
        if mask is not None:
            # 调整 mask 长度以匹配 cnn 输出
            mask = mask[:, : lstm_out.size(1)]
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, L')
        attention_weights = attention_weights.unsqueeze(-1)  # (B, L', 1)

        # 加权求和
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (B, hidden)

        # ========== 分类输出 ==========
        output = self.classifier(context)  # (B, 1)
        return output.squeeze(-1)  # (B,)
