# token_classification

1. 数据处理

   - 数据中的 tokens 是组成一个句子的每个词，如果在 tokenizer 中不传入 is_split_into_words 为 True 的话会将，每个词看成是一个数据，会在每个词编码的前后加入特殊的字符，从而一条数据就生成{"input_ids":[[...],[...]]},但是一条数据应该是{"input_ids":[...]}
   - 由于一个词可能被分成多个字词从而生成多个 ids，word_ids 函数可以生成 ids 对应的 token 的索引，如果是特殊字符的编码那么这个位置便会被赋值成为 None
   - 还是那个问题就是 token 编码完成的 ids 数量是大于等于 ner_tags 的数量的（ner_tags 中的数量应该是和 tokens 中的数量是一致的），所以需要将 ner_tags 和 input_ids 相对应，对应的方法是如果 input_idx 就是一个完整的词没有字词那么就直接拿到 ner_tags 中索引位置，如果一个词被分成多个字词那么只是将第一个字词的部分设置成为索引值，后面字词的位置设置成为-100

2. 加载模型

   - 需要传入 num_labels、id2label 和 label2id

3. compute_metric:
   - 直接拿到的 predictions:[[[embeding] \* tokens] \* batch_size]
   - 转变为[tokens_id] \* batch_size
   - 第一层循环拿到的是每一条数据（batch_size=1）的 ids，l 不等于-100 也会把一个单词的第二个字词给舍弃掉
   - 注意转化成为标签
