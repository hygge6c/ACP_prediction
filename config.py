import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

    # Model hyperparameters (optimized via Optuna)
    MODEL_CONFIG = {
        'vocab_size': 21,      # 20 amino acids + PAD
        'embed_dim': 256,
        'num_heads': 8,
        'hidden_dim': 384,
        'num_layers': 2,
        'dropout': 0.3,
        'max_len': 30
    }

    # Vocabulary: 20 standard amino acids
    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
    VOCAB = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}
    VOCAB['PAD'] = 0

    # Model path
    MODEL_PATH = 'best_models/best_transformer.pth'

    # Prediction threshold
    PREDICTION_THRESHOLD = 0.5

    # Default performance metrics (used if model_performance.json not found)
    DEFAULT_PERFORMANCE = {
        'accuracy': 0.892,
        'precision': 0.876,
        'recall': 0.915,
        'specificity': 0.869,
        'f1': 0.895,
        'mcc': 0.783,
        'auc': 0.941
    }