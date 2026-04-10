import math
import json
import sys
from collections import defaultdict

def train_hmm_trigram(corpus_path, valid_char_set, min_bi_freq = 1, min_tri_freq = 2):
    print("train_hmm_trigram start", file=sys.stderr)
    pi_counts = defaultdict(int)
    bi_counts = defaultdict(lambda: defaultdict(int))
    tri_counts = defaultdict(lambda: defaultdict(int))
    char_occur_counts = defaultdict(int)
    bi_occur_counts = defaultdict(int)
    total_sentences = 0
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()
            if not sentence:
                continue
            total_sentences += 1
            n = len(sentence)
            if n >= 1:
                pi_counts[sentence[0]] += 1
            for i in range(n):
                char_occur_counts[sentence[i]] += 1
                if i < n - 1:
                    c1, c2 = sentence[i], sentence[i+1]
                    bi_counts[c1][c2] += 1
                    bi_occur_counts[c1 + c2] += 1
                if i < n - 2:
                    c1, c2, c3 = sentence[i], sentence[i+1], sentence[i+2]
                    tri_counts[c1 + c2][c3] += 1
    num_chars = len(valid_char_set)
    if num_chars == 0:
        return {}, {}, {}
    default_log_pi = math.log(1 / (total_sentences + num_chars))
    log_pi = {'D': default_log_pi}
    for char, count in pi_counts.items():
        log_pi[char] = math.log((count + 1) / (total_sentences + num_chars))
    log_trans_bi = {}
    for char in valid_char_set:
        occur_count = char_occur_counts.get(char, 0)
        denom = occur_count + num_chars
        char_trans = {}
        has_valid_bi = False
        if char in bi_counts:
            for char2, count in bi_counts[char].items():
                if count >= min_bi_freq:
                    char_trans[char2] = math.log((count + 1) / denom)
                    has_valid_bi = True
        if has_valid_bi:
            char_trans['D'] = math.log(1 / denom)
            log_trans_bi[char] = char_trans
    log_trans_tri = {}
    for c1c2, occur_count in bi_occur_counts.items():
        denom = occur_count + num_chars
        tri_trans = {}
        has_valid_tri = False
        for char3, count in tri_counts[c1c2].items():
            if count >= min_tri_freq:
                tri_trans[char3] = math.log((count + 1) / denom)
                has_valid_tri = True
        if has_valid_tri:
            tri_trans['D'] = math.log(1 / denom)
            log_trans_tri[c1c2] = tri_trans
    return log_pi, log_trans_bi, log_trans_tri

def save_model_json(log_pi, log_trans_bi, log_trans_tri, filepath):
    print("save_model_json start", file=sys.stderr)
    model_data = {
        'pi': log_pi,
        'trans_bi': log_trans_bi,
        'trans_tri': log_trans_tri
    } 
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(
            model_data, 
            f, 
            ensure_ascii=False,
            separators=(',', ':'),
            check_circular=False
        )