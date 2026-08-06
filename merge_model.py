from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

model_name = "LiquidAI/LFM2.5-1.2B-Thinking"
adapter_path = "enseven/lfm-2.5-think-code"
output_dir = "./lfm2.5-model-merged"

base = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base, adapter_path)
merged = model.merge_and_unload()

merged.save_pretrained(output_dir, safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.save_pretrained(output_dir)

print(f"Merged model saved to {output_dir}")
