# translation

1. 数据处理
    - 在tokenizer中传入两个文本的时候对于第二个文本一定要指定text_target这样才会生成labels，否则会将两个文本使用特殊字符（sep）进行分割，导致没有labels

2. data_collator
    - decoder_input_ids和labels之间的关系是pad_token_id进行填充的

3. 训练
    - 在训练的时候要传入prediction_with_generate=True（compute_metrics中的predictions就是idx）

4. 推断
    - 推断的时候要使用generate函数（因为时encoder-decoder架构，输出结果时是decoder）

