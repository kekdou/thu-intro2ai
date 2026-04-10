import tri_src
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAR_TABLE_PATH = os.path.join(BASE_DIR, 'data', '一二级汉字表.txt')
CORPUS_RES_PATH = os.path.join(BASE_DIR, 'output', 'result.txt')
JSON_RES_PATH = os.path.join(BASE_DIR, 'output', 'prob3.json')
PINYIN_MAP_PATH = os.path.join(BASE_DIR, 'data', '拼音汉字表.txt')

def main():
    valid_chars_set = tri_src.load_valid_chars(CHAR_TABLE_PATH)
    log_pi, log_trans_bi, log_trans_tri = tri_src.train_hmm_trigram(CORPUS_RES_PATH, valid_chars_set)
    tri_src.save_model_json(log_pi, log_trans_bi, log_trans_tri, JSON_RES_PATH)
    pinyin = tri_src.Pinyin(JSON_RES_PATH, PINYIN_MAP_PATH)
    for line in sys.stdin:
        pinyin_list = line.strip().split()
        res = pinyin.viterbi_decode(pinyin_list)
        print(res)

if __name__ == "__main__":
    main()