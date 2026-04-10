import torch
import os
import config
from models import get_model
from utils.data_loader import get_dataloader

def test_performance():
    test_x_path = os.path.join(config.DATA_DIR, "test_x.npy")
    test_y_path = os.path.join(config.DATA_DIR, "test_y.npy")
    print(f"--- 正在加载测试数据: {config.MODEL_NAME} ---")
    test_loader = get_dataloader(test_x_path, test_y_path, batch_size=config.BATCH_SIZE, shuffle=False)
    embedding_weight = torch.load(config.EMBEDDING_WEIGHT)
    model = get_model(
        config.MODEL_NAME, 
        embedding_weight, 
        **config.MODEL_CONFIG[config.MODEL_NAME]
    ).to(config.DEVICE)
    model_weight_path = os.path.join(config.SAVE_DIR, f"best_{config.MODEL_NAME.lower()}.pth")
    if not os.path.exists(model_weight_path):
        print(f"错误: 未找到模型权重文件 {model_weight_path}")
        return
    model.load_state_dict(torch.load(model_weight_path, map_location=config.DEVICE))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"模型 [{config.MODEL_NAME}] 测试结果:")
    print(f"测试集准确率 (Test Accuracy): {100 * correct / total:.2f}%")

if __name__ == "__main__":
    test_performance()