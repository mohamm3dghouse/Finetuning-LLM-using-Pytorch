from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
from dotenv import load_dotenv
import torch
import os

def load_peft_model(
    model_id: str,
    hf_token_env_key: str = "HF_TOKEN", 
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: str = "all-linear",
    load_in_8bit: bool = True
):
    # Load environment and authenticate
    load_dotenv()
    hf_token = os.getenv(hf_token_env_key)
    if hf_token is None:
        raise ValueError(f"{hf_token_env_key} not found in .env file.")
    login(token=hf_token)

    # BitsAndBytes config for quantization
    quant_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # torch_dtype = torch.float16,
        # device_map = 'auto',
        quantization_config=quant_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"\n[INFO] Memory footprint of quantized model: {model.get_memory_footprint()/1e9:.2f} GB")

    # Apply LoRA
    peft_config = LoraConfig(
        task_type='CAUSAL_LM',
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    peft_model = get_peft_model(model, peft_config)

    print(f"[INFO] Memory after LoRA wrapping: {peft_model.get_memory_footprint()/1e9:.2f} GB")
    peft_model.print_trainable_parameters()

    return peft_model, tokenizer

