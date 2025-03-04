# 快速开始

## Pipeline

- 只传入一个任务类型

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
# 如果传入的只有一条那么就传入字符串，如果是多条的话就都放在一个列表中
results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
```

- 传入特定的model和tokenizer

```python
from transformers import AutoTokenizer, AutoModel, pipeline
model_name = "your/model/path"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model, tokenizer)
```

## AutoClass

### AutoTokenizer

已经做过详细的介绍

### AutoModel

transformers提供了一个简单统一的方法来对模型进行加载，可以像使用AutoTokenizer一样使用AutoModel，唯一需要注意的是，我们需要为不同的任务选择不同的加载类，比如对于文本分类，需要选择`AutoModelForSequenceClassification`

默认情况下会使用torch.float32进行模型的加载，如果传入参数torch_type='auto'，那么就会按照模型中的配置文件来确定模型的精度



### save model

```python
save_path = 'your/model/path'
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
```



## Custom model builds

- 修改预训练模型中的特定参数

  ```python
  from transformers import AutoConfig
  
  my_config = AutoConfig.from_pretrained("your/model/path", n_heads=12)
  ```

- 加载模型

  ```python
  from transformers import AutoModel
  
  my_model = AutoModel.from_config(my_config)
  ```

  

## Trainer

1. 加载一个预训练模型

   ```python
   from transformers import AutoModel
   model = AutoModel.from_pretrained("your/model/path")
   ```

2. 定义`TrainingArguments`，包含了模型训练时候的超参数

   ```python
   from transformers import TrainingArguments
   # 如果没有传入的话将会使用默认值
   training_args = TrainingArguments(
       output_dir="path/to/save/folder/",
       learning_rate=2e-5,
       per_device_train_batch_size=8,
       per_device_eval_batch_size=8,
       num_train_epochs=2,
   )
   ```

3. 加载处理器

   ```python
   from transformers import AutoTokenizer
   
   tokenizer = AutoTokenizer.from_pretrained("your/tokenizer/path")
   ```

4. 加载数据集

   ```python
   from datasets import load_dataset
   dataset = load_dataset("dataset/name")
   ```

5. 定义处理函数并且将数据集中的元素转化成为数值数据

   ```python
   def tokenize_dataset(example):
       # 这里返回的是数据集中一条元素要被tokenize的元素
       return ....
   dataset = dataset.map(tokenize_dataset, batched=True)
   ```

6. 对数据进行填充，填充后模型可以使用

   ```python
   from transformers import DataCollatorWithPadding
   data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
   ```

7. 定义Trainer并且开始训练

   ```python
   from transformers import Trainer
   
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=dataset["train"],
       eval_dataset=dataset["test"],
       processing_class=tokenizer,
       data_collator=data_collator,
   )
   # 开始对模型进行训练
   trainer.train()
   ```





# 教程

## 使用Piplines进行推导

