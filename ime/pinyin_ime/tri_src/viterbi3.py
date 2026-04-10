import json
from .map import load_pinyin_dict

class Pinyin:
    def __init__(self, model_path, pinyin_dict_path):
        with open(model_path, 'r', encoding='utf-8') as f:
            model = json.load(f)
            self.log_pi = model.get('pi', {})
            self.log_trans_bi = model.get('trans_bi', {})
            self.log_trans_tri = model.get('trans_tri', {})
        self.pinyin_map = load_pinyin_dict(pinyin_dict_path)

    def get_bi_prob(self, char1, char2):
        return self.log_trans_bi.get(char1, {}).get(char2, self.log_trans_bi.get(char1, {}).get('D', -1000))

    def get_tri_prob(self, char1, char2, char3):
        c1c2 = char1 + char2
        if c1c2 in self.log_trans_tri:
            return self.log_trans_tri[c1c2].get(char3, self.log_trans_tri[c1c2].get('D', -1000))
        return self.get_bi_prob(char2, char3) - 2.0

    def viterbi_decode(self, pinyin_list):
        n = len(pinyin_list)
        if n == 0:
            return ""
        dp = [{} for _ in range(n)]
        path = [{} for _ in range(n)]
        candidate0 = self.pinyin_map.get(pinyin_list[0], [])
        for c0 in candidate0:
            dp[0][(None, c0)] = self.log_pi.get(c0, self.log_pi.get('D', -1000))
            path[0][(None, c0)] = None
        if n == 1:
            if not dp[0]: return ""
            return max(dp[0], key=dp[0].get)[1]
        candidate1 = self.pinyin_map.get(pinyin_list[1], [])
        for char1 in candidate1:
            for (none, c0), prob0 in dp[0].items():
                trans_prob = self.get_bi_prob(c0, char1)
                total_prob = prob0 + trans_prob
                state = (c0, char1)
                if state not in dp[1] or total_prob > dp[1][state]:
                    dp[1][state] = total_prob
                    path[1][state] = None 
        for f in range(2, n):
            cur_candidate = self.pinyin_map.get(pinyin_list[f], [])
            for cur_c in cur_candidate:
                for (c_prev_prev, c_prev), prev_prob in dp[f-1].items():
                    trans_prob = self.get_tri_prob(c_prev_prev, c_prev, cur_c)
                    total_prob = prev_prob + trans_prob
                    state = (c_prev, cur_c)
                    if state not in dp[f] or total_prob > dp[f][state]:
                        dp[f][state] = total_prob
                        path[f][state] = c_prev_prev
        if not dp[-1]:
            return ""
        best_state = max(dp[-1], key=dp[-1].get)
        result = [best_state[1], best_state[0]]
        prev_prev = path[n-1][best_state]
        for f in range(n-1, 1, -1):
            result.append(prev_prev)
            state = (prev_prev, result[-2])
            prev_prev = path[f-1][state]
        return "".join(reversed(result))