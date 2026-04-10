import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "processed")
TRAIN_X = os.path.join(DATA_DIR, "train_x.npy")
TRAIN_Y = os.path.join(DATA_DIR, "train_y.npy")
VAL_X = os.path.join(DATA_DIR, "val_x.npy")
VAL_Y = os.path.join(DATA_DIR, "val_y.npy")
WORD_TO_ID = os.path.join(DATA_DIR, "word_to_id.pkl")
EMBEDDING_WEIGHT = os.path.join(DATA_DIR, "embedding_weight.pth")

SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
MAX_LEN = 100

# 可选: 'CNN', 'RNN', 'MLP', 'ATTENTION', 'TRANSFORMER'
MODEL_NAME = 'TRANSFORMER'

MODEL_CONFIG = {
    'CNN': {
        'num_classes': 2, 
        'filter_sizes': [3, 4, 5],
        'num_filters': 100,
        'dropout': 0.5
    },
    'RNN': {
        'num_classes': 2, 
        'hidden_size': 128,
        'num_layers': 2,
        'bidirectional': True,
        'dropout': 0.3
    },
    'MLP': {
        'num_classes': 2, 
        'hidden_dims': [256, 128],
        'dropout': 0.2
    },
    'ATTENTION': {
        'num_classes': 2, 
        'hidden_size': 128, 
        'num_layers': 2, 
        'dropout': 0.3
    },
    'TRANSFORMER': {
        'n_heads': 5,
        'n_layers': 2,
        'dropout': 0.1
    }
}