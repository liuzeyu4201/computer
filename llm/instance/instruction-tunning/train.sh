
root_dir="/root/autodl-tmp"


# 使用一个主机的两个GPU进行模型的训练
# nohup torchrun --nproc_per_node=2 /root/autodl-tmp/code/test.py \
#     --model_path "$root_dir/model/qwen1.5" \
#     --data_path "$root_dir/dataset/alpaca_data_cleaned.json" \
#     --output_dir "$root_dir/code/checkpoint" \
#     --logging_dir "$root_dir/code/log" \
#     --save_path "$root_dir/save_model" \
#     > train.log 2>&1 &


torchrun --nproc_per_node=2 /root/autodl-tmp/code/finetune.py \
    --model_path "$root_dir/model" \
    --data_path "$root_dir/datasets/alpaca_data_cleaned.json" \
    --output_dir "$root_dir/code/checkpoint" \
    --logging_dir "$root_dir/code/log" \
    --save_path "$root_dir/save_model" \


# python /root/autodl-tmp/code/test.py \
#     --model_path "$root_dir/model" \
#     --data_path "$root_dir/datasets/alpaca_data_cleaned.json" \
#     --output_dir "$root_dir/code/checkpoint" \
#     --logging_dir "$root_dir/code/log" \
#     --save_path "$root_dir/save_model"