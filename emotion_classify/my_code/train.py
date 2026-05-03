import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.metrics import f1_score

from models import get_model
from utils.data_loader import get_dataloader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

EPOCHS = 15
MAX_LEN = 100

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = sum([1 if p == l else 0 for p, l in zip(all_preds, all_labels)]) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1

def train_model(model_name, **cfg):
    train_loader = get_dataloader(TRAIN_X, TRAIN_Y, cfg['batch_size'])
    val_loader = get_dataloader(VAL_X, VAL_Y, cfg['batch_size'], shuffle=False)
    embedding_weight = torch.load(EMBEDDING_WEIGHT)
    print(f"--- 正在初始化模型: {model_name.upper()} ---")
    model_cfg = {k: v for k, v in cfg.items() if k not in ['batch_size', 'learning_rate']}
    model = get_model(
        model_name,
        embedding_weight,
        **model_cfg
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = cfg['learning_rate'])
    current_save_path = os.path.join(SAVE_DIR, f"best_{model_name.lower()}.pth")
    best_cal_f1 = 0.0
    print(f"--- 开始训练 (设备: {DEVICE}) ---")
    for _ in range(EPOCHS):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
        val_acc, val_f1 = evaluate(model, val_loader)
        epoch_loss = total_loss / len(train_loader)
        print(f"Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2%} | Val F1: {val_f1:.2%}")
        if val_f1 > best_cal_f1:
            best_cal_f1 = val_f1
            torch.save(model.state_dict(), current_save_path)