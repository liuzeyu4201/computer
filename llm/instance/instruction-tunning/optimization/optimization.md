# notice

## 代码优化
上层文件夹中的代码是可以进行运行的，也就是说可以进行训练的，但是对于数据处理的部分还可以进行优化

### problem
1. 上层代码对于数据的处理流程：首先加载到数据集，然后组装模型需要的prompt(使用了map)，然后将组装的prompt给到tokenizer(这里再次使用了map函数)，由于数据集包括了训练集和测试集，所以这里会map的次数是4。
2. 还有一个地方需要注意的是代码中columns_to_remove需要手动指定，这样不具有一般性（不同的数据集需要进行不同的指定）

注:问题1和labels的组成方式有关，本示例中是将labels中instruction + input编码后的位置设置成为-100，后面labels的编码（小于最大长度）才是数据集中labels的编码。有的方法是prompt直接是instruction + input + output，所以labels直接就会copy tokenize(prompt)的结果，这种方式就不需要第二次的map。

### 解决
1. 对于第一次生成模型所需要的prompt是必须要使用map函数的，对于第二次的map可以借助transformers中提供的数据处理器DataCollatorForSeq2Seq（对于上面注中的第二种方法，当然第一种方法也可以使用Trainer中的data_collator来确保批量长度的一致）
2. 问题出在什么地方？因为调用generate_prompt的时候不仅返回了instruction而且还返回了labels，而labels是模型所需要的，所以dataset.column_names中会有labels，这就会导致不能直接使用**dataset.map(tokenize, remove_columns=dataset.column_names)**。所以我们在对datasets进行处理的时候应该要确保不要直接返回模型所需要的字段，而是要包装到一起或者返回一个其他的字段

## learn
1. 在使用transformers中进行单GPU和多GPU训练的时候只需要控制ddp_find_unused_parameters的数值(多卡False)，不需要添加其他的代码
2. datasets peft Trainer的组合使用

# feedback
如果代码或者表述有问题的欢迎各位大佬提出讨论，我们会对确实存在的错误进行改正
