from transformers import Qwen2Tokenizer, Qwen2ForCausalLM
from peft import PeftModel, PeftConfig

base_model_path = "/root/autodl-tmp/model"
checkpoint = "/root/autodl-tmp/code/checkpoint/checkpoint-1000"

base_model = Qwen2ForCausalLM.from_pretrained(base_model_path)
tokenizer = Qwen2Tokenizer.from_pretrained(base_model_path)

peft_config = PeftConfig.from_pretrained(checkpoint)

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(base_model, checkpoint)

input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

