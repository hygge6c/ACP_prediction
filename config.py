import os


class Config:
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'

    # 模型配置
    MODEL_CONFIG = {
        'vocab_size': 21,  # 20个氨基酸 + PAD
        'embed_dim': 256,  # 从Optuna获得
        'num_heads': 8,  # 从Optuna获得
        'hidden_dim': 384,  # 从Optuna获得
        'num_layers': 2,  # 从Optuna获得
        'dropout': 0.3,  # 从Optuna获得
        'max_len': 30
    }

    # 模型路径
    MODEL_PATH = 'best_models/best_transformer_1.pth'

    # 氨基酸字典
    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
    VOCAB = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
    VOCAB['PAD'] = 0

    # 预测阈值
    PREDICTION_THRESHOLD = 0.5

    # 性能指标
    PERFORMANCE_METRICS = {
        'accuracy': 0.892,
        'precision': 0.876,
        'recall': 0.915,
        'specificity': 0.869,
        'f1': 0.895,
        'mcc': 0.783,
        'auc': 0.941
    }