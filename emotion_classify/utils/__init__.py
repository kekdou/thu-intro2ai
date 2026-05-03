import os
import torch
import numpy as np
import pickle

from .preprocess import TextPreprocs
from .embed import build_embedding_weight
from .csv2txt import process_csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "Dataset", "train.txt")
VAL_PATH = os.path.join(BASE_DIR, "Dataset", "validation.txt")
TEST_PATH = os.path.join(BASE_DIR, "Dataset", "test.txt")
STOP_PATH = os.path.join(BASE_DIR, "Dataset", "hit_stopwords.txt")
W2V_PATH = os.path.join(BASE_DIR, "Dataset", "wiki_word2vec_50.bin")

SAVE_PATH = os.path.join(BASE_DIR, "processed")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

CSV_PATH = os.path.join(BASE_DIR, "Dataset", "ChnSentiCorp_htl_all.csv")
TXT_PATH = os.path.join(BASE_DIR, "Dataset", "ChnSentiCorp_htl_all.txt")

MAX_LEN = 100
MIN_FREQ = 2

def save_processed_data(save_dir, x_data, y_data, prefix):
    np.save(os.path.join(save_dir, prefix + "_x.npy"), x_data)
    np.save(os.path.join(save_dir, prefix + "_y.npy"), y_data)

def check_data_exists():
    essential_files = [
        os.path.join(SAVE_PATH, "train_x.npy"),
        os.path.join(SAVE_PATH, "train_y.npy"),
        os.path.join(SAVE_PATH, "val_x.npy"),
        os.path.join(SAVE_PATH, "val_y.npy"),
        os.path.join(SAVE_PATH, "test_x.npy"),
        os.path.join(SAVE_PATH, "test_y.npy"),
        os.path.join(SAVE_PATH, "embedding_weight.pth"),
        os.path.join(SAVE_PATH, "word_to_id.pkl")
    ]
    return all(os.path.exists(f) for f in essential_files)

def check_csv_exists():
    essential_files = [
        os.path.join(SAVE_PATH, "csv_x.npy"),
        os.path.join(SAVE_PATH, "csv_y.npy"),
    ]
    return all(os.path.exists(f) for f in essential_files)

def initialize_data(force_rebuild=False):
    if not force_rebuild and check_data_exists():
        print(f"数据已处理，存于 {SAVE_PATH}，跳过初始化。")
        return
    
    print("开始执行数据初始化...")
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    processor = TextPreprocs(max_len=MAX_LEN, min_freq=MIN_FREQ)
    if os.path.exists(STOP_PATH):
        processor.load_stop_words(STOP_PATH)

    train_labels, train_texts = processor.read_data(TRAIN_PATH)
    processor.build_vocab(train_texts)
    processor.save_vocab(SAVE_PATH)
    train_x = processor.text_to_sequence(train_texts)
    train_y = np.array(train_labels)
    
    val_labels, val_texts = processor.read_data(VAL_PATH)
    val_x = processor.text_to_sequence(val_texts)
    val_y = np.array(val_labels)

    test_labels, test_texts = processor.read_data(TEST_PATH)
    test_x = processor.text_to_sequence(test_texts)
    test_y = np.array(test_labels)

    save_processed_data(SAVE_PATH, train_x, train_y, "train")
    save_processed_data(SAVE_PATH, val_x, val_y, "val")
    save_processed_data(SAVE_PATH, test_x, test_y, "test")

    embedding_weight = build_embedding_weight(processor.word_to_id, W2V_PATH)
    torch.save(embedding_weight, os.path.join(SAVE_PATH, "embedding_weight.pth"))

    print(f"数据初始化完成！")

def initialize_csv():
    if check_csv_exists():
        return
    if not os.path.exists(CSV_PATH):
        print(f"{CSV_PATH} 文件不存在")
        return
    if not os.path.exists(TXT_PATH):
        process_csv(CSV_PATH, TXT_PATH)

    vocab_path = os.path.join(SAVE_PATH, "word_to_id.pkl")
    processor = TextPreprocs(max_len=MAX_LEN, min_freq=MIN_FREQ)
    with open(vocab_path, "rb") as f:
        processor.word_to_id = pickle.load(f)
    processor.id_to_word = {v: k for k, v in processor.word_to_id.items()}
    if os.path.exists(STOP_PATH):
        processor.load_stop_words(STOP_PATH)
    
    csv_labels, csv_texts = processor.read_data(TXT_PATH)
    csv_x = processor.text_to_sequence(csv_texts)
    csv_y = np.array(csv_labels)
    save_processed_data(SAVE_PATH, csv_x, csv_y, "csv")
    print(f"初始化完成！")
