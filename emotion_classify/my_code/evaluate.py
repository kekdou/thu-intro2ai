import torch
import os

from models import get_model
from utils.data_loader import get_dataloader
from sklearn.metrics import precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "processed")
EMBEDDING_WEIGHT = os.path.join(DATA_DIR, "embedding_weight.pth")

SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_performance(model_name, force_csv=False, **cfg):
    x_file = "csv_x.npy" if force_csv else "test_x.npy"
    y_file = "csv_y.npy" if force_csv else "test_y.npy"
    test_x_path = os.path.join(DATA_DIR, x_file)
    test_y_path = os.path.join(DATA_DIR, y_file)

    print(f"--- 正在加载测试数据: {model_name.upper()} ---")
    test_loader = get_dataloader(test_x_path, test_y_path, batch_size=cfg['batch_size'], shuffle=False)
    embedding_weight = torch.load(EMBEDDING_WEIGHT)
    model_cfg = {k: v for k, v in cfg.items() if k not in ['batch_size', 'learning_rate']}
    model = get_model(
        model_name,
        embedding_weight,
        **model_cfg
    ).to(DEVICE)
    model_weight_path = os.path.join(SAVE_DIR, f"best_{model_name.lower()}.pth")
    if not os.path.exists(model_weight_path):
        print(f"错误: 未找到模型权重文件 {model_weight_path}")
        return
    model.load_state_dict(torch.load(model_weight_path, map_location=DEVICE))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = sum([1 if p == l else 0 for p, l in zip(all_preds, all_labels)]) / len(all_labels)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"模型 [{model_name.upper()}] 测试结果:")
    print(f"{'-'*30}")
    print(f"Accuracy:  {acc:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-Score:  {f1:.2%}")
    print(f"{'-'*30}")
