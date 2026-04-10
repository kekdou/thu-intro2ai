import re
import os
import ast
import sys

SEP_RE = re.compile(r'[^\u4e00-\u9fa5]+')

def load_valid_chars(filepath):
    """
    加载一二级汉字表，加入到集合中并返回
    """
    print("load_valid_chars start", file=sys.stderr)
    if not os.path.exists(filepath):
        return set()
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        valid_chars = {c for line in f for c in line if not c.isspace()}
    return valid_chars

def clean_and_filter_text(text, valid_chars_set):
    """
    基于一二级汉字表对 text 进行过滤
    并且以非汉字为分隔，对 text 切片
    """
    if not text:
        return []
    raw_pieces = SEP_RE.split(text)
    final_sentences = []
    for piece in raw_pieces:
        if piece:
            cleaned = ''.join(filter(valid_chars_set.__contains__, piece))
            final_sentences.append(cleaned)
    return final_sentences

def process_corpus(input_path, output_file, encoding, file_type, valid_chars_set): 
    """
    处理语料，目前仅包括 weibo 和 sina
    """
    print("process_corpus start", file=sys.stderr)
    if os.path.isdir(input_path):
        for file_name in os.listdir(input_path):
            file_path = os.path.join(input_path, file_name)
            process_corpus(file_path, output_file, encoding, file_type, valid_chars_set)
        return
    if not os.path.exists(input_path):
        return 
    valid_line_count = 0
    with open(input_path, 'r', encoding=encoding, errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = ast.literal_eval(line)
                raw_sentences = []
                if file_type == 'weibo':
                    raw_sentences.append(data.get('content', ''))
                elif file_type == 'sina':
                    raw_sentences.append(data.get('title', ''))
                    raw_sentences.append(data.get('html', ''))
                for text in raw_sentences:
                    cleaned_list = clean_and_filter_text(text, valid_chars_set)
                    for sentence in cleaned_list:
                        output_file.write(sentence + '\n')
                        valid_line_count += 1
            except Exception as e:
                continue