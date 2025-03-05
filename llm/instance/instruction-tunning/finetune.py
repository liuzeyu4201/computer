import os
import argparse
import torch
import torch.distributed as dist
from transformers import Trainer, TrainingArguments, Qwen2Tokenizer, Qwen2ForCausalLM
from peft import get_peft_model, LoraConfig
from process import prepare_full_train_test_dataset
from transformers import DataCollatorForSeq2Seq

def get_args():
    parser = argparse.ArgumentParser(description="参数传入")
    parser.add_argument("--model_path", type=str, help="模型的位置")
    parser.add_argument("--data_path", type=str, help="数据位置")
    parser.add_argument("--output_dir", type=str, help="中间过程的checkpoint输出位置")
    parser.add_argument("--logging_dir", type=str, help="训练中的日志输出位置")
    parser.add_argument("--save_path", type=str, help="训练完成之后模型的存储位置")
    return parser.parse_args()

if __name__ == '__main__':
    args_parser = get_args()
    model_path = args_parser.model_path
    data_file = args_parser.data_path
    # local_rank = setup_ddp()

    train_dataset, test_dataset = prepare_full_train_test_dataset(data_file, model_path)
    print(f"训练的第一条数据{train_dataset[0]}")
    base_model = Qwen2ForCausalLM.from_pretrained(model_path)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, config)
    model.print_trainable_parameters()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name} is trainable")

    train_args = TrainingArguments(
        output_dir=args_parser.output_dir,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        logging_dir=args_parser.logging_dir,
        logging_steps=50,
        logging_strategy="steps",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="steps",
        eval_steps=1000,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        load_best_model_at_end=True,
        # fp16=True,
        report_to="none",
        # 多卡训练选项
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        # data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(args_parser.save_path)