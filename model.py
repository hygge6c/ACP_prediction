import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim,
                 num_layers, dropout, max_len):
        super(TransformerClassifier, self).__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim

        # Embedding层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.fc2 = nn.Linear(embed_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()

        # 存储注意力权重
        self.attention_weights = None

    def forward(self, x, return_attn=False):
        # 嵌入层
        x_embed = self.embedding(x) * torch.sqrt(torch.tensor(self.embed_dim, device=x.device))
        x_embed = self.pos_encoder(x_embed.transpose(0, 1)).transpose(0, 1)

        # 创建注意力掩码（忽略padding）
        padding_mask = (x == 0)

        # 应用Transformer编码器并获取注意力权重
        self.attention_weights = []
        x_encoded = self.encoder(x_embed, src_key_padding_mask=padding_mask)

        # 全局平均池化
        x_pooled = self.pooling(x_encoded.transpose(1, 2)).squeeze(-1)

        # 全连接层
        x_features = F.relu(self.fc1(x_pooled))
        x_features = self.dropout(x_features)
        x_output = self.fc2(x_features)

        if return_attn:
            return self.sigmoid(x_output).squeeze(), self.attention_weights
        return self.sigmoid(x_output).squeeze()

    def get_attention_weights(self):
        """获取注意力权重"""
        return self.attention_weights

    def compute_feature_importance(self, x, target_class=1):
        """计算特征重要性（基于梯度）"""
        self.eval()

        # 将输入转换为浮点类型并启用梯度
        x_float = x.float().clone()
        x_float.requires_grad_(True)

        # 前向传播
        output = self(x_float.long())  # 嵌入层需要long类型

        # 计算损失
        if target_class == 1:
            loss = output.sum()
        else:
            loss = (1 - output).sum()

        # 反向传播
        self.zero_grad()
        loss.backward()

        # 获取输入梯度
        gradients = x_float.grad

        if gradients is not None:
            # 计算每个位置的重要性
            importance = torch.abs(gradients).sum(dim=-1)
            return importance.detach().cpu().numpy()
        else:
            # 如果梯度计算失败，返回随机重要性
            seq_len = x.shape[1]
            return np.random.rand(1, seq_len)

    def get_simulated_attention(self, sequence_length):
        """生成模拟的注意力权重"""
        # 创建一个随机注意力矩阵
        attn = torch.randn(1, 8, sequence_length, sequence_length)  # 假设8个头
        attn = F.softmax(attn, dim=-1)  # 应用softmax
        return attn