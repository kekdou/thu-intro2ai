import numpy as np
import os
import pickle

from collections import Counter

class TextPreprocs:
    def __init__(self, max_len = 100, min_freq = 2):
        self.max_len = max_len
        self.min_freq = min_freq
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.word_to_id = {self.PAD_TOKEN: self.PAD_ID, self.UNK_TOKEN: self.UNK_ID}
        self.id_to_word = {self.PAD_ID: self.PAD_TOKEN, self.UNK_ID: self.UNK_TOKEN}
        self.stopwords = set()

    def load_stop_words(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.stopwords.add(line)

    def read_data(self, file_path):
        labels = []
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    label = int(parts[0])
                    words = parts[1].split()
                    words = [w for w in words if w not in self.stopwords]
                    labels.append(label)
                    texts.append(words)
        return labels, texts
    
    def build_vocab(self, texts):
        word_counts = Counter()
        for words in texts:
            word_counts.update(words)
        idx = len(self.word_to_id)
        for word, count in word_counts.items():
            if count >= self.min_freq:
                self.word_to_id[word] = idx
                self.id_to_word[idx] = word
                idx += 1
        
    def text_to_sequence(self, texts):
        sequences = []
        for words in texts:
            seq = [self.word_to_id.get(w, self.UNK_ID) for w in words]
            if len(seq) > self.max_len:
                seq = seq[:self.max_len]
            else:
                seq = seq + [self.PAD_ID] * (self.max_len - len(seq))
            sequences.append(seq)
        return np.array(sequences)
    
    def save_vocab(self, save_dir):
        vocab_path = os.path.join(save_dir, "word_to_id.pkl")
        with open(vocab_path, "wb") as f:
            pickle.dump(self.word_to_id, f)
        

