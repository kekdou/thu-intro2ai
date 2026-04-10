import sys
import json
import math
import gc

from collections import defaultdict

class PinyinIME:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = []
        self.pinyin_to_ids = defaultdict(list)
        # 一元字计数
        self.unigram_counts = []
        # 二元字计数
        self.bigram_counts = []
        # 总字数
        self.total_unigrams = 0
        # 线性插值因子
        self.lam = 0.9

    def get_id(self, char):
        if char not in self.char_to_id:
            char_id = len(self.id_to_char)
            self.char_to_id[char] = char_id
            self.id_to_char.append(char)
            # 占位，与索引对应
            self.unigram_counts.append(0)
            self.bigram_counts.append({})
            return char_id
        return self.char_to_id[char]

    def load_data(self):
        with open('./word2pinyin.txt', 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                char = parts[0]
                char_id = self.get_id(char)
                for py in parts[1:]:
                    self.pinyin_to_ids[py].append(char_id)
        with open('./1_word.txt', 'r', encoding='utf-8') as f:
            data = json.load(f)
            for py, item in data.items():
                for word, count in zip(item["words"], item["counts"]):
                    if word in self.char_to_id:
                        char_id = self.char_to_id[word]
                        self.unigram_counts[char_id] += count
                        self.total_unigrams += count
        with open('./2_word.txt', 'r', encoding='utf-8') as f:
            data = json.load(f)
            for _, item in data.items():
                for word_pair, count in zip(item["words"], item["counts"]):
                    chars = word_pair.split()
                    if chars[0] in self.char_to_id and chars[1] in self.char_to_id:
                        id0, id1 = self.char_to_id[chars[0]], self.char_to_id[chars[1]]
                        self.bigram_counts[id0][id1] = count
        del data
        gc.collect()

    def get_prob(self, prev_id, curr_id):
        p_unigram = (self.unigram_counts[curr_id] + 1) / (self.total_unigrams + len(self.id_to_char))
        char1 = self.unigram_counts[prev_id]
        if char1 > 0:
            char1_to_char2 = self.bigram_counts[prev_id].get(curr_id, 0)
            p_bigram = char1_to_char2 / char1
            return math.log(self.lam * p_bigram + (1 - self.lam) * p_unigram)
        else:
            return math.log(p_unigram)

    def viterbi(self, pinyin_list):
        candidates = self.pinyin_to_ids.get(pinyin_list[0], [])
        if not candidates:
            return ""
        dp = [{} for _ in range(len(pinyin_list))]
        path = [{} for _ in range(len(pinyin_list))]
        for char_id in candidates:
            prob = (self.unigram_counts[char_id] + 1) / (self.total_unigrams + len(self.id_to_char))
            dp[0][char_id] = math.log(prob)
        for f in range(1, len(pinyin_list)):
            curr_candidates = self.pinyin_to_ids.get(pinyin_list[f], [])
            prev_states = dp[f-1]
            for cur in curr_candidates:
                max_lp = -float('inf')
                best_pre = None
                for prev, prev_lp in prev_states.items():
                    lp = prev_lp + self.get_prob(prev, cur)
                    if lp > max_lp:
                        max_lp = lp
                        best_pre = prev
                if best_pre is not None:
                    dp[f][cur] = max_lp
                    path[f][cur] = best_pre
            if not dp[f]: 
                return ""
        last_layer = dp[-1]
        if not last_layer:
            return ""
        curr_best = max(last_layer, key=last_layer.get)
        result = []
        for f in range(len(pinyin_list) - 1, -1, -1):
            result.append(self.id_to_char[curr_best])
            curr_best = path[f].get(curr_best)
        return "".join(reversed(result))

def main():
    ime = PinyinIME()
    ime.load_data()
    for line in sys.stdin:
        pinyin_list = line.strip().split()
        if pinyin_list:
            print(ime.viterbi(pinyin_list))

if __name__ == "__main__":
    main()