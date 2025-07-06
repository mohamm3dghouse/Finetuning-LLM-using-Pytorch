import torch
import regex as re

def run_eval(df, model, tokenizer, batch_size):
    all_messages = []
    all_targets = []
    df_copy = df.copy()
    df_copy.reset_index(inplace=True)
    batch_size = len(df_copy) if len(df_copy)<=batch_size else batch_size  # Or whatever size you want
    i = 0  # The current batch index (update in your loop)
    # Define the true labels you’ll compare against
    target_categories = ["Robotics", "Machine Learning", "Artificial Intelligence", "Computer Vision", "Discrete Mathematics"]
    # Function to build the prompt
    def build_prompt(abstract):
        return (
            f"Read the following abstract from a scientific paper and guess its research area from the following list:\n\n"
            f"{', '.join(target_categories)}\n\n"
            f"Abstract:\n{abstract}\n\n"
            f"Answer with just the single category name."
        )
    all_outputs = []

    for i in range(0, len(df_copy), batch_size):
        batch_df = df_copy[i : i + batch_size]
        batch_df.reset_index(inplace=True)
        # print(len(batch_df))
    
        # Build messages for each abstract in the batch
        messages_batch = []
        targets_batch = []
    
        for row in range(len(batch_df)):
            prompt = build_prompt( batch_df['abstract'][row])
            
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

        inputs = tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding = True,    
        ).to(model.device)#.to(torch.bfloat16)

        all_messages.extend(messages_batch)
        all_targets.extend(targets_batch)
        
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=25)
        
        outputs = tokenizer.batch_decode(outputs) 
        all_outputs.extend(outputs)
        del inputs
        del outputs
        torch.cuda.empty_cache()
    # print('cleared memory')

    #############################
    #predict
    # break
    predictions = []
    correct = 0
    outputs = all_outputs
    for row in range(len(outputs)):
        # Robust regex: get everything after <start_of_turn>model until <end_of_turn> if it exists
        match = re.search(r"<start_of_turn>model(.*?)(?:<end_of_turn>|$)", outputs[row], re.DOTALL)
    
        guess_raw = match.group(1).strip() if match else None
        # print(response)
        guess = ''
        try:
          # Optional cleanup / normalization
          guess = guess_raw.lower().strip().replace(".", "")
        except:
          guess = guess_raw.lower()
        # print(
              # 'Messages:', messages, '\n\n',
              # 'Outputs:', outputs, '\n\n',
              # 'Guess_raw:', guess_raw,
              # '\n\n',
              # 'Guess:', guess, '\n\n',
              # 'category name:', df_copy['category_name'][row]
              # )

        # Match against expected labels (basic matching)
        matched = None
        for cat in target_categories:
            if cat.lower() in guess:
                matched = cat
                break

        predictions.append(matched or guess_raw)  # fallback to raw guess
    
        # Accuracy check
        if matched == df_copy['category_name'][row]:
            correct += 1

        # break
    
    # Store predictions
    # df_copy["predicted_category"] = predictions
    
    # Accuracy
    accuracy = correct / len(df_copy)
    print(f"\n🎯 Accuracy (exact match with known categories for {len(df_copy)} inputs): {accuracy:.2%}")
