import math
import json
import sys

from collections import defaultdict

def train_hmm(corpus_path, valid_char_set):
    print("train_hmm strat", file=sys.stderr)
    # char 作为句子第一个字出现的次数
    pi_counts = defaultdict(int)
    # char1 转移到 char2 的次数
    trans_counts = defaultdict(lambda: defaultdict(int))
    # char 出现的总次数
    char_occur_counts = defaultdict(int)
    # 句子总数
    total_sentences = 0
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()
            if not sentence:
                continue
            total_sentences += 1
            pi_counts[sentence[0]] += 1
            for i in range(len(sentence) - 1):
                cur_char = sentence[i]
                next_char = sentence[i + 1]
                trans_counts[cur_char][next_char] += 1
                char_occur_counts[cur_char] += 1
            char_occur_counts[sentence[-1]] += 1
    num_chars = len(valid_char_set)
    default_log_pi = math.log(1 / (total_sentences + num_chars))
    log_pi = {'D': default_log_pi}
    for char, count in pi_counts.items():
        # P = (char 开头次数 + 1) / (总句子数 + 汉字总数)
        log_pi[char] = math.log((count + 1) / (total_sentences + num_chars))
    log_trans = {}
    for char in valid_char_set:
        occur_count = char_occur_counts.get(char, 0)
        denom = occur_count + num_chars
        # 默认概率键为 D
        char_trans_data = {'D': math.log(1 / denom)}
        if char in trans_counts:
            for char2, count in trans_counts[char].items():
                char_trans_data[char2] = math.log((count + 1) / denom)
        log_trans[char] = char_trans_data
    return log_pi, log_trans

def save_model_json(log_pi, log_trans, filepath):
    print("save_model_json start", file=sys.stderr)
    model_data = {
        'pi': log_pi,
        'trans': log_trans
    } 
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(
            model_data, 
            f, 
            ensure_ascii=False,
            separators=(',', ':'),
            check_circular=False
        )