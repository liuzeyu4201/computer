# multiple choice

1. 数据处理格式

   - 输入 = 完整句子
   - 输出 = 引导句子 + 各种选择的句子（假设有 n 中选择）
   - 处理：
     - 先将完整句子 copy n 遍 => sent1
     - 将引导句子 copy n 遍并且和 n 中选择相互组成完整的含有选项的答案 => sent2
     - 将 sent1 和 sent2 展开成为一层列表
       - 因为 tokenizer 只能处理一层列表的结构
     - 重新分组：{"input_ids":[[[sentence1 choice1 ids],[sentence2 choice2 ids]....],[[sentence2 choice1 ids],[sentence2 choice2 ids]]]}
       - 使得到的 input_ids 的长度和加载的数据集的数据量是一样的
       - sentcence1 choice1 ids 和 sentence1 choice2 ids 的长度不一定相同，也就是说内部的列表（含有 ids 的列表的长度是不一定相同的）

2. 多项选择题（有固定的答案）可以直接使用 accuracy 作为指标即可

3. 推断时加入到 tokenizer 中的数据是[[prompt, candidate1], [prompt, candidate2]]
