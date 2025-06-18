# text classification

1. 数据处理：
    - 语料中存在labels和text的话，直接将text进行tokenize就行

2. compute_metrics
    - 对于encoder-only架构的模型prediction拿到的就是模型输出的每个词的预测概率

3. 模型加载
    - num_labels、id2label和label2id
