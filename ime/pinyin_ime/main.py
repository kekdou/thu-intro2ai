import src
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAR_TABLE_PATH = os.path.join(BASE_DIR, 'data', '一二级汉字表.txt')
WEIBO_PATH = os.path.join(BASE_DIR, 'corpus', 'SMP2020', 'usual_train_new.txt')
SINA_PATH = os.path.join(BASE_DIR, 'corpus', 'sina_news_gbk')

CORPUS_RES_PATH = os.path.join(BASE_DIR, 'output', 'result.txt')
JSON_RES_PATH = os.path.join(BASE_DIR, 'output', 'prob.json')

PINYIN_MAP_PATH = os.path.join(BASE_DIR, 'data', '拼音汉字表.txt')

def main():
    valid_chars_set = src.load_valid_chars(CHAR_TABLE_PATH)
    if not valid_chars_set:
        return
    with open(CORPUS_RES_PATH, 'w', encoding='utf-8') as out_f:
        src.process_corpus(WEIBO_PATH, out_f, 'utf-8', 'weibo', valid_chars_set)
        src.process_corpus(SINA_PATH, out_f, 'gbk', 'sina', valid_chars_set)
    log_pi, log_trans = src.train_hmm(CORPUS_RES_PATH, valid_chars_set)
    src.save_model_json(log_pi, log_trans, JSON_RES_PATH)
    pinyin = src.Pinyin(JSON_RES_PATH, PINYIN_MAP_PATH)
    for line in sys.stdin:
        pinyin_list = line.strip().split()
        res = pinyin.viterbi_decode(pinyin_list)
        print(res)

if __name__ == '__main__':
    main()