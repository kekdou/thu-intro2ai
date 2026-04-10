import json

from .map import load_pinyin_dict

class Pinyin:
    def __init__(self, model_path, pinyin_dict_path):
        """
        初始化训练结果和拼音表
        """
        with open(model_path, 'r', encoding='utf-8') as f:
            model = json.load(f)
            self.log_pi = model['pi']
            self.log_trans = model['trans']
        self.pinyin_map = load_pinyin_dict(pinyin_dict_path)

    def viterbi_decode(self, pinyin_list):
        """
        接收拼音串 pinyin_list 返回输入法输出
        """
        dp = []
        path = []
        first_pinyin = pinyin_list[0]
        first_candidate = self.pinyin_map.get(first_pinyin, [])
        f0_dp = {}
        f0_path = {}
        for char in first_candidate:
            f0_dp[char] = self.log_pi.get(char, self.log_pi.get('D', -1000))
            f0_path[char] = None
        dp.append(f0_dp)
        path.append(f0_path)
        for f in range(1, len(pinyin_list)):
            cur_pinyin = pinyin_list[f]
            cur_candidates = self.pinyin_map.get(cur_pinyin, [])
            cur_f_dp = {}
            cur_f_path = {}
            for cur_char in cur_candidates:
                max_prob = -float('inf')
                best_prev = None
                for prev_char, prev_prob in dp[f - 1].items():
                    trans_prob = self.log_trans.get(prev_char, {}).get(cur_char, 
                                 self.log_trans.get(prev_char, {}).get('D', -1000))
                    total_prob = prev_prob + trans_prob
                    if total_prob > max_prob:
                        max_prob = total_prob
                        best_prev = prev_char
                cur_f_dp[cur_char] = max_prob
                cur_f_path[cur_char] = best_prev
            dp.append(cur_f_dp)
            path.append(cur_f_path)
        if not dp[-1]:
            return ""
        best_char = max(dp[-1], key=dp[-1].get)
        result = []
        for f in range(len(pinyin_list) - 1, -1, -1):
            result.append(best_char)
            best_char = path[f][best_char]
        return "".join(reversed(result))
