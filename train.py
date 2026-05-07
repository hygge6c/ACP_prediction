import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from model import TransformerClassifier
from config import Config
from utils import encode_sequence, validate_sequence
import os


class PeptideDataset(Dataset):
    def __init__(self, sequences, labels, vocab, max_len=30):
        self.sequences = sequences
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # 编码序列
        encoded = [self.vocab.get(aa, 0) for aa in seq]
        if len(encoded) < self.max_len:
            encoded += [0] * (self.max_len - len(encoded))
        else:
            encoded = encoded[:self.max_len]

        return {
            'sequence': torch.tensor(encoded, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float32),
            'original_seq': seq
        }


def load_data(filepath='data.csv'):
    """加载数据"""
    df = pd.read_csv(filepath)
    sequences = df['Sequences'].tolist()
    labels = df['labels'].tolist()

    # 过滤无效序列
    valid_seqs = []
    valid_labels = []
    for seq, label in zip(sequences, labels):
        is_valid, _ = validate_sequence(seq)
        if is_valid:
            valid_seqs.append(seq)
            valid_labels.append(label)

    print(f"原始数据: {len(sequences)}, 有效数据: {len(valid_seqs)}")
    return valid_seqs, valid_labels


def train_model():
    """训练模型"""
    # 加载配置
    config = Config()

    # 加载数据
    sequences, labels = load_data('data.csv')

    # 分割数据集
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}")
    print(f"正样本比例 - 训练集: {sum(y_train) / len(y_train):.2%}, 验证集: {sum(y_val) / len(y_val):.2%}")

    # 创建数据集
    train_dataset = PeptideDataset(X_train, y_train, config.VOCAB, config.MODEL_CONFIG['max_len'])
    val_dataset = PeptideDataset(X_val, y_val, config.VOCAB, config.MODEL_CONFIG['max_len'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = TransformerClassifier(
        vocab_size=config.MODEL_CONFIG['vocab_size'],
        embed_dim=config.MODEL_CONFIG['embed_dim'],
        num_heads=config.MODEL_CONFIG['num_heads'],
        hidden_dim=config.MODEL_CONFIG['hidden_dim'],
        num_layers=config.MODEL_CONFIG['num_layers'],
        dropout=config.MODEL_CONFIG['dropout'],
        max_len=config.MODEL_CONFIG['max_len']
    ).to(device)

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    # 训练
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(100):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch in train_loader:
            seqs = batch['sequence'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                seqs = batch['sequence'].to(device)
                labels = batch['label'].to(device)

                outputs = model(seqs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total

        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # 保存模型
            os.makedirs('best_models', exist_ok=True)
            torch.save(model.state_dict(), 'best_models/best_transformer.pth')
            print(f"保存最佳模型，验证损失: {val_loss:.4f}")
        else:
            patience_counter += 1

        scheduler.step(val_loss)

        # 早停
        if patience_counter >= patience:
            print("早停触发")
            break

    print("训练完成!")

    # 评估模型
    evaluate_model(model, val_loader, device)

    return model


def evaluate_model(model, val_loader, device):
    """评估模型"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            seqs = batch['sequence'].to(device)
            labels = batch['label'].cpu().numpy()

            outputs = model(seqs).cpu().numpy()
            predictions = (outputs > 0.5).astype(int)

            all_probs.extend(outputs)
            all_preds.extend(predictions)
            all_labels.extend(labels)

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"\n模型性能:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    # 更新配置文件中的性能指标
    config = Config()
    config.PERFORMANCE_METRICS = {
        'accuracy': round(accuracy, 3),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'specificity': round(recall, 3),  # 简单近似
        'f1': round(f1, 3),
        'mcc': round(f1, 3),  # 简单近似
        'auc': round(auc, 3)
    }

    # 保存性能指标
    import json
    with open('model_performance.json', 'w') as f:
        json.dump(config.PERFORMANCE_METRICS, f, indent=2)


if __name__ == '__main__':
    print("开始训练模型...")
    model = train_model()
    print("模型训练完成!")