import torch
from datasets import load_dataset
from transformers import LlamaTokenizer

# 参数
CUTOFF_LEN = 256


def split_train_test(data_path):
    """
        划分训练集和测试集（先打乱数据再切分，以避免数据分布不均）
    """
    try:
        dataset = load_dataset("json", data_files=data_path, split="train")
        dataset = dataset.shuffle()
        datadict = dataset.train_test_split(test_size=0.2)
        return datadict["train"], datadict["test"]
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None


def prepare_tokenizer(tokenize_path):
    """
        初始化 tokenizer
    """
    tokenizer = LlamaTokenizer.from_pretrained(tokenize_path, add_eos_token=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    tokenizer.padding=True
    return tokenizer


def generate_prompt(data_point):
    """
        生成文本格式的 prompt生成prompt
    """
    if data_point["input"]:
        instruction = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            ### Instruction:
            {data_point["instruction"]}
            ### Input:
            {data_point["input"]}
            ### Response:\n
    """
    else:
        instruction = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
            ### Instruction:
            {data_point["instruction"]}
            ### Response:\n
    """

    return {'instruction': instruction, 'labels':data_point["output"]}


def generate_model_input(tokenizer, dataset,MAX_LENGTH=CUTOFF_LEN):
    """
        将文本数据转换为 tokens, dataset是拼接完成的prompt
    """
    def tokenize(data_point):
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer(data_point["instruction"],padding='max_length',max_length=CUTOFF_LEN)
        response = tokenizer(data_point["labels"] + tokenizer.eos_token,padding='max_length',max_length=CUTOFF_LEN)
        input_ids = instruction["input_ids"] + response["input_ids"]
        attention_mask = instruction["attention_mask"] + response["attention_mask"]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]

        return {
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'labels':labels
        }
    return dataset.map(tokenize, remove_columns=["instruction", "input", "output"])


def prepare_full_train_test_dataset(data_file, tokenizer_path):
    train_dataset, test_dataset = split_train_test(data_file)
    tokenizer = prepare_tokenizer(tokenizer_path)
    train_prompt_dataset = train_dataset.map(generate_prompt)
    test_prompt_dataset = test_dataset.map(generate_prompt)
    train_dataset = generate_model_input(tokenizer, train_prompt_dataset)
    test_dataset = generate_model_input(tokenizer, test_prompt_dataset)
    return train_dataset, test_dataset
"""

函数使用流程：
if __name__ == '__main__':
    data_file = "/root/autodl-tmp/dataset/alpaca_data_cleaned.json"
    tokenize_path = "/root/autodl-tmp/model"
    # 划分训练集和测试集
    train_dataset, test_dataset = split_train_test(data_file)
    # 初始化 tokenizer
    tokenizer = prepare_tokenizer(tokenize_path)
    # 生成 prompt 数据集
    print(f"正在生成模型所需的prompt")
    full_prompt_dataset = train_dataset.map(generate_prompt)
    # 生成 tokens 数据集
    print("正在生成模型编码")
    dataset = generate_model_input(tokenizer, full_prompt_dataset)
    # 打印测试数据
    print(f"查看测试数据：{dataset[0]}")


"""