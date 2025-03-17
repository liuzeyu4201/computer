# summarization

1. 数据处理
    - 在seq2seq的任务中tokenize的时候为了方便理解输入的sequence还是输出的sequence(labels)，对于tokenizer中输出的sequence部分前要加入参数text_target以区分，如果tokenizer中传入的是两个变量，那么text_target部分会转化成为labels

2. data_collator
    - 对于seq2seq的任务在创建data_collator的时候需要传入model
    - data_collator会将之前已经处理过的数据中的labels向右偏移一位生成decoder_input_ids

3. 评估
    - 要将-100替换成为tokenizer.pad_token_id，这样在使用decode函数解码时不会报错
    - 评估函数的用法， 这里的评估函数不需要长度相等

4. 训练
    - 使用的时Seq2SeqTrainingArguments和Seq2SeqTrainer
    - 还需要在Seq2SeqTrainingArgument设置predict_with_generate为True（这样在compute_metrics中的predictions中可以直接拿到idx）

5. 推断
    - 由于seq2seq模型使用的是encoder-decoder架构，所以在生成结果时用的时decoder生成的，所以要使用generate函数
