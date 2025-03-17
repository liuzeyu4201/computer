# Casual language modeling


1. 数据处理格式（分组）

   - 将 batch 中的所有的目标字符串拼接在一起之后 tokenize 该数据
     - 如果是先 tokenizer 后拼接特殊的字符会有点问题，如果 tokenize 的数据前后都会添加特殊字符就不对了
   - 设置每组的最长可包含的 ids 的长度，input_ids 是一层列表
   - 将得到的所有的 ids 按照最大长度进行截取（如果不能整除就舍去后面不能整除的部分）
   - 得到的结果:{"input_ids":[[...],[...]...]}, len([...]) = group_max_length

2. tokenizer 设置
   由于是要生成句子的 ids，所以如果遇到填充的时候就应该停止，所以设置

   ```
   tokenizer.pad_token = tokenizer.eos_token
   ```

3. data_collator 设置
   在设置 data_collator 的时候需要传入 mlm=False:自回归模型的 labels 是 input_ids 的副本向右偏移一位，在 DataCollatorForLanguageModeling 中传入 mlm 后，data_collator 在处理数据的时候会自动生成 labels，labels 是 input_ids 向右偏移的一位
