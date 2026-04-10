import torch
import torch.nn as nn
import torch.optim as optim
import os

import config
from models import get_model
from utils.data_loader import get_dataloader
from utils import initialize_data

def evaluate(model, loder):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in loder:
            texts, labels = texts.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def train():
    train_loader = get_dataloader(config.TRAIN_X, config.TRAIN_Y, config.BATCH_SIZE)
    val_loader = get_dataloader(config.VAL_X, config.VAL_Y, config.BATCH_SIZE, shuffle=False)
    embedding_weight = torch.load(config.EMBEDDING_WEIGHT)
    print(f"--- 正在初始化模型: {config.MODEL_NAME} ---")
    model = get_model(
        config.MODEL_NAME, 
        embedding_weight, 
        **config.MODEL_CONFIG[config.MODEL_NAME]
    ).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = config.LEARNING_RATE)
    current_save_path = os.path.join(config.SAVE_DIR, f"best_{config.MODEL_NAME.lower()}.pth")
    best_cal_acc = 0.0
    print(f"--- 开始训练 (设备: {config.DEVICE}) ---")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_acc = evaluate(model, val_loader)
        print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2%}")
        if val_acc > best_cal_acc:
            best_cal_acc = val_acc
            torch.save(model.state_dict(), current_save_path)

if __name__ == "__main__":
    initialize_data()
    train()