import pandas as pd
import regex as re
import torch
from transformers import AutoTokenizer
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Function to build the prompt
def build_prompt(abstract, target_categories):
    return (
        f"Read the following abstract from a scientific paper and guess its research area from the following list:\n\n"
        f"{', '.join(target_categories)}\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"Answer with just the single category name."
    )


def build_dataset(df, tokenizer, categories):
    """build dataset batch by batch"""
    batch_size = 100  # Or whatever size you want
    i = 0  # The current batch index (update in your loop)
    
    # Define the true labels you’ll compare against
    target_categories = list(categories.values())
    
    all_messages = []
    all_targets = []
    all_raw_targets = []

    for i in range(0, len(df), batch_size):
        batch_df = df[i : i + batch_size]
        
        batch_df.reset_index(inplace=True)
        # Build messages for each abstract in the batch
        messages_batch = []
        targets_batch = []
        raw_targets_batch = []
    
        for row in range(len(batch_df)):
            prompt = build_prompt(batch_df['abstract'][row], target_categories)
        
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
            messages_batch.append(messages)
       
            targets = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                },
                {
                    "role": "model",
                    "content": [{"type": "text", "text": batch_df['category_name'][row]}]
                }
            ]
            
            raw_targets_batch.append({'messages':targets}) # useful when using SFT trainer
    
            targets = tokenizer.apply_chat_template(
                targets,
                add_generation_prompt=False,
                continue_final_message=True,
                tokenize=False,
                # return_dict=True,
                # return_tensors="pt",
                padding = True
            ) + '.' + '<end_of_turn>' #tokenizer.eos_token      
            targets_batch.append(targets)

        all_messages.extend(messages_batch)
        all_targets.extend(targets_batch)
        all_raw_targets.extend(raw_targets_batch)

    torch.cuda.empty_cache()
    return all_messages, all_targets, all_raw_targets


def get_batch(all_messages:list, all_targets: list, tokenizer: AutoTokenizer, device: str='cpu', ):
    targets_tensor = tokenizer(
    all_targets,
    return_tensors="pt",
    padding = True,
    add_special_tokens=False
    )['input_ids'].to(device)

    inputs = tokenizer.apply_chat_template(
    all_messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    padding = True,
        ).to(device)
    
    input_tensor = inputs['input_ids']

    mask = torch.ones_like(input_tensor, dtype=torch.bool).to(input_tensor.device)

    pad_len = targets_tensor.size(1) - input_tensor.size(1)  # difference in width
    mask = torch.cat([mask, torch.zeros(mask.size(0), pad_len, dtype=torch.bool).to(device=input_tensor.device)], dim=1)

    # Masking prompt tokens
    targets_tensor_masked = targets_tensor.masked_fill(mask, torch.tensor(-100))

    # Shifting targets for next token prediction
    targets_tensor_masked_shifted = targets_tensor_masked[:,1:]

    return targets_tensor[:,:-1], targets_tensor_masked_shifted

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN not found in .env file.")
    login(token=hf_token)

    model_id = "google/gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load dataset
    data_json = "arxiv_dataset.json"
    if not os.path.isfile(data_json):
        raise FileNotFoundError("Required file not found: Run ~get_data.py~ first")

    df = pd.read_json("arxiv_dataset.json")
    df.rename(columns={"summary": "abstract"}, inplace=True)
    # Shuffle the dataset to ensure randomness
    df = df.sample(frac=1).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    df_train, df_test = df[:split_idx], df[split_idx:]

    print(len(df_train), len(df_test))

    # Category code to human-readable name
    categories = {
        "cs.RO": "Robotics",
        "cs.LG": "Machine Learning",
        "cs.AI": "Artificial Intelligence",
        "cs.CV": "Computer Vision",
        "cs.DM": "Discrete Mathematics"
    }

    all_messages, all_targets, all_raw_targets = build_dataset(
                                                                df_train,
                                                                tokenizer,
                                                                categories
                                                                )
    
    print(len(all_messages), len(all_targets), len(all_raw_targets))
