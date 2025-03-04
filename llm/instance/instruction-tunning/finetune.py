import argparse
import transformers
from evaluate import load
import numpy as np
from transformers import Trainer, TrainingArguments
from peft import (
    get_peft_model,
    LoraConfig
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from Data_Process import prepare_full_train_test_dataset

def get_args():
    parser = argparse.ArgumentParser(description="参数传入")
    parser.add_argument("--model_path", type=str, help="模型的位置")
    parser.add_argument("--data_path", type=str, help ="数据位置")
    parser.add_argument("--output_dir", type=str, help ="中间过程的checkpoint输出位置")
    parser.add_argument("--logging_dir", type=str, help="训练中的日志输出位置")
    parser.add_argument("--save_path", type=str, help="训练完成之后模型的存储位置")
    return parser.parse_args()

def setup_ddp():
    # 初始化分布式
    dist.init_process_group(backend="nccl")  # GPU 通信方式
    local_rank = int(os.environ["LOCAL_RANK"])  # 获取当前进程的 GPU 编号
    torch.cuda.set_device(local_rank)
    return local_rank



if __name__ == '__main__':
    args_parser = get_args()
    model_path = args_parser.model_path
    data_file = args_parser.data_path
    local_rank = setup_ddp()
    train_dataset, test_dataset = prepare_full_train_test_dataset(data_file, model_path)
    base_model =  LlamaForCausalLM.from_pretrained(model_path).to(local_rank)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    config = LoraConfig(
        r=8,    # 较低的秩可以降低模型的计算量，但是会影响模型的效果
        lora_alpha=16,  # 较大的lora_alpha会增加低秩矩阵的影响，可能会提高模型的表达能力，但可能会导致过拟合
        target_modules=["q_proj","v_proj"], # 模型的哪些模块使用Lora
        lora_dropout=0.1,   # 可以防止过拟合
        bias="none",    # 默认情况下是None
        task_type="CAUSAL_LM"   # 任务类型
    )
    # PEFT模型
    model = get_peft_model(base_model, config).to(local_rank)

    
    train_args = TrainingArguments(
        output_dir=args_parser.output_dir,          # 微调模型之后保存的文件夹
        save_strategy="steps",               # 保存模型策略
        save_steps=500,                     # 500step保存一次
        save_total_limit=3,                 # 最多保存3个
        logging_dir=args_parser.logging_dir,               # 日志保存的文件夹
        logging_steps=500,                  # 打印日志的步数
        logging_strategy="steps",            # 日志打印策略
        per_device_train_batch_size=2,     # 训练时批量大小
        per_device_eval_batch_size=4,   # 评估时批量大小
        eval_strategy="steps",              # 评估策略
        eval_steps=500,                   # 评估步数
        gradient_accumulation_steps=4,      # 梯度累计步数
        num_train_epochs=3,                 # 训练多少轮
        learning_rate=5e-5,                 # 学习率
        warmup_steps=100,                   # 预热步数
        weight_decay=0.01,                  # 梯度衰减
        load_best_model_at_end=True,        # 加载最好的模型
        fp16=True,                           # 启动混合精度训练
        report_to="none"

    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        # compute_metrics=compute_metrics
        # 使用ddp多卡训练
        # ddp_find_unused_parameters=False,
    )


    # if local_rank == 0:
    #     print("准备开始训练")
    trainer.train()
    # trainer.save_model()  # 保存模型  # 这种存储会将所有的模型都保存，相对来说会占用比较大的空间
    save_path = args_parser.save_path
    # if local_rank == 0:
        # trainer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    # trainer.save_state()  # 保存训练状态（例如优化器状态）
    # dist.destroy_process_group()

