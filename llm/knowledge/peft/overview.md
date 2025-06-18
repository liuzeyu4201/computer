# 简单使用

- 模型训练

```python
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, TaskType, get_peft_model

###############
# 这里只关注下面的几行代码，也就是peft简单使用
peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-large")
model = get_peft_model(model, peft_config)
###############

training_args = TrainingArguments(...)
training_args = TrainingArguments(...)
model.save_pretrained("your/path")
```

- 模型推断

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

model = AutoPeftModelForCausalLM.from_pretrained("your/path")
tokenizer = AutoTokenizer.from_pretrained("model match tokenizer")

model = model.to("cuda")
model.eval()
inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])
```



# 综述

## 使用步骤

- **目的**：快速使用高效微调方法微调模型
- **如何实现目的**：得到peftModel
- **如何得到peftModel**：需要两个组件，一个方法
  - 高效微调配置文件
  - 原始的模型
  - get_peft_model方法

- **便捷之处**：可以直接和transformers等库相兼容

## 微调方法

微调方法也就是说我们要如何指定配置文件，首先需要先知道有peft支持哪几种微调方法

- Prompt-base methods
- Lora  methods
- IA3



# 主要对象

## Configuration

**配置类的组织方式**：

- PeftConfigMixin
  - PeftConfig
    - PromptLearningConfig（soft prompt方法的父类）
    - LoraConfig
    - ...

**注意**：什么方法需要传入什么参数去官网进行查看

## PEFT types

包含peft支持的适配器（adapters ）以及peft支持的任务类型

|     适配器      |            任务类型            |
| :-------------: | :----------------------------: |
|  PROMPT_TUNING  |      SEQ_CLS（文本分类）       |
|    P_TUNING     | SEQ_2_SEQ_LM（语言到语言建模） |
|  PREFIX_TUNING  |   CAUSAL_LM（因果关系模型）    |
|      LORA       |     TOKEN_CLS（标记分类）      |
| ADAPTION_PROMPT |    QUESTION_ANS（问题回答）    |
|       ...       | FEATURE_EXTRACTION（特征提取） |



## PEFT model

这个对象是我们使用get_peft_model函数得到的peft_model对象

### 子类

- PeftModelForSequenceClassification
- PeftModelForTokenClassification
- PeftModelForCausalLM
- PeftModelForSeq2SeqLM
- PeftModelForFeatureExtraction



### 属性

|           属性            |           作用            |         限制         |
| :-----------------------: | :-----------------------: | :------------------: |
|           model           | 返回基础的transformer模型 |          无          |
|        peft_config        |      peft模型的配置       |          无          |
|      modules_to_save      |  要保存的子模块列表名称   |          无          |
|      prompt_encoder       |       prompt编码器        | PromptLearningConfig |
|       prompt_tokens       |    虚拟的prompt token     | PromptLearningConfig |
| transformer_backbone_name |  基础模型中主干模型名称   | PromptLearningConfig |
|      word_embeddings      |    基础模型中的词嵌入     | PromptLearningConfig |

### 方法

- **`add_adapter(adapter_name: str, peft_config: PeftConfig,low_cpu_mem_usage: bool = False)`**
  - 向模型中添加peft_config的适配器
  - 新添加的这个适配器是没有经过任何训练的
  - adapter_name应该是唯一的
  - 默认添加的这个适配器是没有自动被激活的，需要使用set_adapter()函数激活
- **`load_adapter(model_id, adapter_name, is_trainable)`**
  - 将一个训练好的适配器放入模型
  - 这个适配器的名字不能和模型中已经存在的适配器名字一样
- **`set_adapter(adapter_name)`**
  - 将名字为adapter_name的适配器激活
  - 一次只能有一个适配器处于激活的状态
- **`save_pretrained(save_directory)`**
  - 将适配器模型以及适配器的配置文件存储到目标文件夹
- **`from_pretrained(model, adapter_name, is_trainable)`**
  - 使用peft权重去实例化一个预训练模型
  - 原始的预训练模型结构可能会发生变化

- **`disable_adapter()`**
  - 上下文管理器去禁止使用适配器
  - 在这个上下文管理器中model使用的是原始的model
- **`get_base_model()`**
  - 返回原始模型
- **`get_nb_trainable_parameters()`**
  - 返回模型可训练参数的数量以及模型参数数量
- **`print_trainable_parameters()`**
  - 返回模型中的可训练参数
  - 和get_nb_trainable_parameters()方法返回的有所差别
- **`get_prompt(batch_size)`**
  - 返回peft模型使用的虚拟prompt
  - 只有在使用prompt learning的时候才可以使用
- **`get_prompt_embedding_to_save()`**
  - 返回存储模型是要保存的prompt编码
  - 只有在使用prompt learning的时候才可以使用

- **`get_model_status()`**
  - 获取模型优化器状态
- **`get_layer_status()`**
  - 获取模型中每个适配器层的状态

注意：还有一个**PeftMixedModel**方法没有介绍，他是一种特殊的peft模型，可以混合不同类型的适配器



## Utilities

- **`peft.cast_mixed_precision_params(model, dtype)`**
  - 将模型的所有不可训练参数转换为给定的数据类型

- **`peft.get_peft_model(model, peft_config, adapter_name:str="default")`**
  - 从基础模型和配置中得到一个peft模型

- **`peft.inject_adapter_in_model(peft_config, model, adapter_name)`**
  - 创造适配器并且将其注入到模型中
  - 不支持prompt learning 和 adaption prompt

- **`peft.get_peft_model_state_dict(model)`**
  - 返回peft模型的状态字典

- **`peft.prepare_model_for_kbit_training(model)`**
  - 这个方法仅是支持transformers模型

- **`peft.get_layer_status(model)`**
  - 获取模型优化器状态

- **`peft.get_model_status(model)`**
  - 获取模型中每个适配器层的状态
