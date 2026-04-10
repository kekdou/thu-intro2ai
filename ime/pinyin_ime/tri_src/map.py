def load_pinyin_dict(filepath):
    pinyin_map = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            pinyin = parts[0]
            chars = parts[1:]
            pinyin_map[pinyin] = chars
    return pinyin_map