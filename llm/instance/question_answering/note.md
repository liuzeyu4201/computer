# question answering

**注意一定要知道数据处理要处理成为什么样子的，肯定是要和input_ids对应上，因为只有input_ids模型才能用，这个task中还需要input_ids的起始位置和结束位置**

一定要的数据：context:str, question:str, answers:{"text":[str], "answer_start"}

1. 由于一个id（一个context包含一个text, answer_start, 所以直接使用flatten整平数据

2. 训练的tokenize
    - 和直觉上有一个较大的差别是，传入tokenizer的两个参数，第一个是question，第二个是context，和我们平常生活中的有点差别

    - 由于context可能是很长的，所以在传入trunction的时候要传入only_second,而且由于我们想要的答案可能在trunction的后面吗，传入stride可以将一条context切份成为好几个数据（不是按照顺序，而是随机的），会造成的一个现象就是数据集的条数增多了

    - 要将return_offsets_mapping和return_overflowing_tokens设置成为True
        - offset（以一条数据为例：不batch的话需要再次嵌套列表）[(ids1_char_start, ids2_char_end),(ids2_char_start, ids2_char_end)...],特殊的字符不是列表而是(0,0)，也就是说offset是ids和char的一个映射
    
    - 使用answer中提供的起始字符以及context可以拿到char_start, end_start

    - tokenized_data.sequence_ids可以拿到ids是传入的第一个文本中的内容，还是第二个文本那种的内容，如果是特殊字符会返回None

    - 接下来就是要将char和idx进行映射了，为了防止过的循环，这里指定了start_index和end_idx,表示文本在offset中的起始位置和结束位置，这样就不用循环问题的部分

    - 根据context中的offset和答案的其实字符和结束字符确定答案在context中的起始idx和末尾idx，并且放入start_positions和end_positions中

3. 验证的tokenize = > 主要是为了在compute_metrics中使用，因为没有开始索引和结束索引
   
   
    **为什么需要example_id**:因为tokenizer中传入的stride导致的id对应不上的问题
    - 在验证集上不能知道答案的起始位置，所以需要将答案的偏移量设置成为None
    - tokenizer之后的overflow_to_sample_mapping和函数中自带传入的examples可以得到原始的id，overflow_to_sample_mapping中的长度和input_ids的长度是一致的，但是应该大于等于examples中的长度

4. compute_metrics函数 => tokenize后的validation作用体现
    - features由于是经过tokenizer处理过的，所以条数大于等于examples中的条数，又因为start_logits和end_logits是由features生成的，所以他们两个的条数应该是和features的条数是一样的
    - 统将有着相同example_id的数据索引用列表统计起来{"id":[0,1,2]...},如果某个id对应的id中只有一条元素那么他就是没有受到tokenizer中stride的影响
    - 对于真实数据中的一条数据，取到被切分成多个input_ids中的最大分数，所以需要两层循环，内部循环是处理一条数据由于stride切分成多条数据的情况，外部循环时为了处理examples中所有的条数
    - 内层的start_indexes循环和end_indexes都是stride可能造成影响的条数