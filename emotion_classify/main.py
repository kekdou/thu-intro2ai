import sys
import config
from my_code import train_model
from my_code import test_performance
from utils import initialize_data
from utils import initialize_csv

def error_print():
    print("请输入 python main.py [task] [model_name] [other_check]")
    print("pragram: 1. train\t 2. evaluate")
    print("model_name: 1. mlp\t 2. cnn\t 3. rnn\t 4. attention\t 5. transformer")
    print("other_check(可选): 1. y\t 2. n")

def main():
    if len(sys.argv) < 3:
        error_print()
        return
    task = sys.argv[1]
    model_name = sys.argv[2].lower()
    if model_name not in ['mlp', 'cnn', 'rnn', 'attention', 'transformer']:
        error_print()
        return

    model_name = model_name.upper()
    csv_check = len(sys.argv) >= 4 and sys.argv[3].lower() == 'y'
    cfg = config.MODEL_CONFIG[model_name]
    
    initialize_data()
    if task == 'train':
        train_model(model_name, **cfg)
    elif task == 'evaluate':
        if csv_check:
            initialize_csv()
        test_performance(model_name, csv_check, **cfg)
    else:
        error_print()


if __name__ == "__main__":
    main()

