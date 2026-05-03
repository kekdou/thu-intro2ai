## 文本情感分类

### 数据集使用 

课程下发的 `train.txt`, `validation.txt`, `test.txt`, `wiki_word2vec_50.bin`  

此外还有检验鲁棒性所用数据集 `ChnSentiCorp_htl_all.csv`   

以及配合数据初始化的停词表 `hit_stopwords.txt`

### 使用方法

1. 安装依赖

```shell
pip install -r requirements.txt
```

2. 训练模型

```shell
python main train [option]
```
option 包括实现的模型
- mlp
- cnn
- rnn
- attention
- transformer

例如训练 cnn 模型

```shell
python main.py train cnn
```

得到类似以下的输出：

```
开始执行数据初始化...
数据初始化完成！
--- 正在初始化模型: CNN ---
--- 开始训练 (设备: gpu) ---
Loss: 0.5078 | Val Acc: 81.28% | Val F1: 81.23%
Loss: 0.3484 | Val Acc: 83.80% | Val F1: 83.77%
Loss: 0.2479 | Val Acc: 84.60% | Val F1: 84.57%
Loss: 0.1639 | Val Acc: 85.11% | Val F1: 85.11%
......
```

3. 测试模型

```shell
python main.py evaluate [option] 
```

option 选项与前一步相同  

可能的输出：

```
数据已处理，存于 path/processed，跳过初始化。
--- 正在加载测试数据: CNN ---
模型 [CNN] 测试结果:
------------------------------
Accuracy:  85.64%
Precision: 85.67%
Recall:    85.66%
F1-Score:  85.64%
------------------------------
```

其中可以选择使用其他数据集来测试鲁棒性  

```shell
python main.py evaluate [option] y
```

```
初始化完成！
--- 正在加载测试数据: CNN ---
模型 [CNN] 测试结果:
------------------------------
Accuracy:  79.74%
Precision: 76.68%
Recall:    75.36%
F1-Score:  75.94%
------------------------------
```

### 文件结构

```
2024010091-wangyk
├── checkpoints/                    # 存放模型参数
├── Dataset/
│   ├── ChnSentiCorp_htl_all.csv    # 酒店评论文件
│   ├── hit_stopwords.txt           # 停词表
│   ├── test.txt
│   ├── train.txt
│   ├── validation.txt
│   └── wiki_word2vec_50.bin
├── models/
│   ├── __init__.py
│   ├── attention.py
│   ├── cnn.py
│   ├── mlp.py
│   ├── rnn.py
│   └── transformer.py
├── my_code/
│   ├── __init__.py 
│   ├── evaluate.py 
│   └── train.py
├── processed/                      # 存放处理好的数据
├── Dataset/
│   ├── __init__.py
│   ├── csv2txt.py
│   ├── data_loader.py
│   ├── embed.py
│   └── preprocess.py
├── README.md
├── main.py 
├── requirements.txt
└── config.py             
```