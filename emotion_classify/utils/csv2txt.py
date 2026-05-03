import pandas as pd
import jieba

def process_csv(csv_filepath, txt_filepath):
    df = pd.read_csv(csv_filepath)
    df['review'] = df['review'].fillna("").astype(str)
    texts = df['review'].tolist()
    labels = df['label'].tolist()
    labels = [0 if l == 1 else 1 for l in labels]
    prcs_texts = []
    for text in texts:
        words = jieba.lcut(text)
        prcs_texts.append(words)
    with open(txt_filepath, 'w', encoding='utf-8') as f:
        for label, text in zip(labels, prcs_texts):
            f.write(f"{label}\t{' '.join(text)}\n")
