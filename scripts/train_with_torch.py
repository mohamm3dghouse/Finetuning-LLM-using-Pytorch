import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from run_eval import run_eval
from preprocess_data import build_dataset, get_batch
import pandas as pd
from model import load_peft_model
import os
from get_data import categories
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
import gc

# ------------------------------
# Configs
# ------------------------------
MODEL_ID = "google/gemma-3-1b-it"
DATA_JSON = "arxiv_dataset.json"
EPOCHS = 2
BATCH_SIZE = 2
ACCUMULATION_STEPS = 2
EVAL_INTERVAL = 100
EVAL_BATCH_SIZE = 20
LEARNING_RATE = 5e-5
# ------------------------------
# Helper Functions
# ------------------------------
def prepare_dataset(tokenizer):
    # get data
    if not os.path.isfile(DATA_JSON):
        raise FileNotFoundError("Required file not found: Run ~get_data.py~ first")

    print(f"[Info] Loading dataset from: {DATA_JSON}")
    df = pd.read_json(DATA_JSON)
    df.rename(columns={"summary": "abstract"}, inplace=True)

    # Shuffle the dataset to ensure randomness
    print("[Info] Shuffling and splitting dataset...")
    df = df.sample(frac=1).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    df_train, df_test = df[:split_idx], df[split_idx:]
    ###
    print(f"[Info] Dataset split -> Training: {len(df_train)} Samples | Testing: {len(df_test)} Samples")

    print("Paper categories: ", list(categories.values()))

    print("[Info] Building tokenized dataset...")
    all_messages, all_targets, all_raw_targets = build_dataset(
                                                                df_train,
                                                                tokenizer,
                                                                categories
                                                                )

    print(f"[Info] Tokenized -> Messages: {len(all_messages)} | Targets: {len(all_targets)} | Raw Targets: {len(all_raw_targets)}")
    return df_train, df_test, all_messages, all_targets


def calulate_loss(logits, targets):
    return F.cross_entropy(logits, targets, ignore_index=-100)


def train_model(model, tokenizer, all_messages, all_targets, df_test):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)#, weight_decay=0.01)

    # Effective batch size = BATCH_SIZE * ACCUMULATION_STEPS
    for epoch in range(EPOCHS):
        print(f'Epoch: {epoch}')
        
        loop = tqdm(range(int(len(all_targets)/BATCH_SIZE)), desc=f"At epoch{epoch}")
        
        # for i in range(int(len(all_targets)/batch)):
        for i in loop:
            inp, tar = get_batch(
                            all_messages[BATCH_SIZE * i: BATCH_SIZE * (i + 1)],
                            all_targets[BATCH_SIZE * i: BATCH_SIZE * (i + 1)],
                            tokenizer,
                            model.device
                        )

        
            out = model(inp)
            B, T, logits = out.logits.shape
            tar = tar.reshape(-1)
            loss = calulate_loss(out.logits.view(B*T, -1), tar)
    
            # Normalize loss for accumulation
            loss = loss / ACCUMULATION_STEPS        
            # Backward pass
            loss.backward()        
            # Accumulate gradients
            if (i + 1) % ACCUMULATION_STEPS == 0:
                # Update weights
                optimizer.step()
                optimizer.zero_grad()
            
            loop.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)  # Update tqdm with the current "loss"

            loss.detach()
            del inp
            del out
            del tar
            del loss
            torch.cuda.empty_cache()
            # break
            if (((i) * BATCH_SIZE)) % 100 == 0:
                run_eval(df_test[:100], model, tokenizer, 20)
                torch.cuda.empty_cache()

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    print(f"[Info] Loading model and tokenizer from: {MODEL_ID}")
    model, tokenizer = load_peft_model(MODEL_ID)

    df_train, df_test, all_messages, all_targets = prepare_dataset(tokenizer)

    print("[Eval] Running initial evaluation on test set...")
    run_eval(df_test[:100], model, tokenizer, 20)
    gc.collect()
    torch.cuda.empty_cache()

    train_model(model, tokenizer, all_messages, all_targets, df_test)

