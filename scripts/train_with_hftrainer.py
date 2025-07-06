import os
import gc

import torch
import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments 

from run_eval import run_eval
from preprocess_data import build_dataset
from model import load_peft_model
from get_data import categories

# ------------------------------
# Configs
# ------------------------------
MODEL_ID = "google/gemma-3-1b-it"
DATA_JSON = "arxiv_dataset.json"
EPOCHS = 2
BATCH_SIZE = 1
ACCUMULATION_STEPS = 2
EVAL_INTERVAL = 50
EVAL_BATCH_SIZE = 2
LEARNING_RATE = 5e-5

# ------------------------------
# Helper Functions
# ------------------------------
def prepare_dataset(tokenizer, split='train'):
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

    if split=='train':
        all_messages, all_targets, all_raw_targets = build_dataset(
                                                                    df_train,
                                                                    tokenizer,
                                                                    categories
                                                                    )
    else:
        all_messages, all_targets, all_raw_targets = build_dataset(
                                                                    df_test,
                                                                    tokenizer,
                                                                    categories
                                                                    )


    print(f"[Info] Tokenized -> Messages: {len(all_messages)} | Targets: {len(all_targets)} | Raw Targets: {len(all_raw_targets)}")
    return df_train, df_test, all_messages, all_targets, all_raw_targets


def create_hf_dataset(all_raw_targets):
    dataset = Dataset.from_list(all_raw_targets)

    def tokenize_and_mask(examples):
        input_ids_batch = []
        attn_mask_batch = []
        labels_batch = []
        # print(len(examples["messages"]))
        # print(examples["messages"])
        for messages in examples["messages"]:
            # print()
            input_ids_batch.append(messages)
            attn_mask_batch.append(messages)
            labels_batch.append(messages)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attn_mask_batch,
            "labels": labels_batch
        }

    # Apply tokenization and masking
    tokenized_dataset = dataset.map(
        tokenize_and_mask,
        batched=True,
        batch_size=2,
        remove_columns=["messages"]
    )
    return tokenized_dataset
    # Set format for PyTorch
    # tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


def data_collater(examples):
    input_ids_list = []
    labels_list = []

    for ex in examples:
        messages = ex["input_ids"]

        prompt_msgs = messages[:-1]
        full_msgs = messages

        prompt_str = tokenizer.apply_chat_template(
            prompt_msgs,
            add_generation_prompt=True,
            tokenize=False
        )
        full_str = tokenizer.apply_chat_template(
            full_msgs,
            add_generation_prompt=False,
            continue_final_message=True,
            tokenize=False
        ) + '.'+'<end_of_turn>'
        # print('inside \n\n',full_str)
        # Tokenize separately without padding
        prompt_tokens = tokenizer(
            prompt_str,
            add_special_tokens=False
        )
        full_tokens = tokenizer(
            full_str,
            add_special_tokens=False
        )

        input_ids = full_tokens["input_ids"]

        labels = input_ids.copy()
        prompt_len = len(prompt_tokens["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len

        # input_ids_list.append({"input_ids": input_ids[:-1]})
        # labels_list.append({"input_ids": labels[1:]}) #hf trainer internally handles label shifting
        input_ids_list.append({"input_ids": input_ids[:]})
        labels_list.append({"input_ids": labels[:]})

    # Now dynamically pad
    batch_input = tokenizer.pad(
        input_ids_list,
        padding=True,
        return_tensors="pt"
    )
    batch_labels = tokenizer.pad(
        labels_list,
        padding=True,
        return_tensors="pt"
    )
    batch_labels["input_ids"][batch_labels["input_ids"] == tokenizer.pad_token_id] = -100
    # print(batch_input["input_ids"].shape)
    # print(batch_labels["input_ids"].shape)
    return {
        "input_ids": batch_input["input_ids"],
        "attention_mask": batch_input["attention_mask"],
        "labels": batch_labels["input_ids"]
    }


def train_model(tokenised_train_dataset, tokenised_eval_dataset):
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        logging_dir="./logs",
        logging_steps=2,  # <-- log every n steps
        logging_strategy="steps",  # Ensure logging is done per step (not per epoch)
        disable_tqdm=False,       # Show progress bar and logs in notebook
        report_to="none",
        label_names=["labels"],  # Explicitly specify the label names
        gradient_accumulation_steps = ACCUMULATION_STEPS,
        learning_rate = LEARNING_RATE,
        torch_empty_cache_steps = 1,
        eval_strategy = "steps",
        eval_steps = EVAL_INTERVAL,
        per_device_eval_batch_size  = EVAL_BATCH_SIZE,
        # accelerator_config = {'split_batches ':True}
    )

    # model = AutoModelForSeq2SeqLM.from_pretrained("google/gemma-2b-it")  # example model

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_train_dataset,
        eval_dataset = tokenised_eval_dataset,
        data_collator= data_collater,
    )

    trainer.train()

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    print(f"[Info] Loading model and tokenizer from: {MODEL_ID}")
    model, tokenizer = load_peft_model(MODEL_ID)

    df_train, df_test, all_messages, all_targets, all_raw_targets = prepare_dataset(tokenizer, split='train')
    tokenised_train_dataset = create_hf_dataset(all_raw_targets=all_raw_targets)

    df_train, df_test, all_messages, all_targets, all_raw_targets = prepare_dataset(tokenizer, split='test')
    tokenised_eval_dataset = create_hf_dataset(all_raw_targets=all_raw_targets)

    print("[Eval] Running initial evaluation on test set...")
    run_eval(df_test[:100], model, tokenizer, 20)
    gc.collect()
    torch.cuda.empty_cache()

    train_model(tokenised_train_dataset, tokenised_eval_dataset)
