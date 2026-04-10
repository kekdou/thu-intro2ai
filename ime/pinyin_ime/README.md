## 拼音输入法

### 语料情况
使用 微博 `usual_train_new.txt` 和 新浪新闻 `sina_news_gbk` 中的语料

### 使用

二元模型：

```shell
python main.py < ./data/input.txt > ./data/output.txt
```

三元模型

```shell
python main3.py < ./data/input.txt > ./data/output.txt
```

若不想替换原先输出结果，可将 `output.txt` 更改为其他，如 `output3.txt`

### 文件目录

```
2024010091-wangyk
├── data/
│   ├── 拼音汉字表
│   ├── 一二级汉字表.txt   
│   ├── input.txt
|   ├── output.txt         # 二元模型生成
│   └── answer.txt
├── src/
│   ├── __init__.py
│   ├── corpus.py
│   ├── hmm.py
│   ├── map.py
│   └── viterbi.py
├── tri_src/
│   ├── __init__.py
│   ├── corpus.py
│   ├── hmm3.py
│   ├── map.py
│   └── viterbi3.py
├── output/
│   ├── result.txt        # 预处理语料
│   ├── prob.json         # 二元模型概率表
│   ├── prob3.json        # 三元模型概率表
├── README.md
├── main.py               # 使用二元模型
└── main3.py              # 使用三元模型
```