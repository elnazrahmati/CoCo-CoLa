from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from datasets import Dataset
import pandas as pd
import torch
import os
import argparse

def calculate_accuracy(model, tokenizer, device, df):
    total, correct = 0, 0
    preds = []
    df['prompt'] = [f"### Question: {q}\n ### Answer: " for q in df['question']]
    for i in range(0, len(df), 128):
        inputs = tokenizer(df['prompt'][i:i+128].tolist(), return_tensors='pt', padding=True, truncation=True).to(device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = model.generate(input_ids, attention_mask=attention_mask, do_sample=False, max_new_tokens=20)
        preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))


    df['prediction'] = preds

    df['answer'] = df['answer'].astype(str)

    for i in range(len(df)):
        refs = df['answer'][i].split(' ')
        pred = df['prediction'][i]
        for ref in refs:
            if ref.strip().lower() in pred.strip().lower():
                correct += 1
                break
        total += 1


    return correct / total, df



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_language', type=str, default='english')
    parser.add_argument('--language', type=str, default='english')
    parser.add_argument('--finetuned', type=int, default=1)
    parser.add_argument('--model_size', type=int, default=8)
    
    args = parser.parse_args()

    if args.model_size == 1:
        lang_checkpoints = {
            'english': 300, 
            'hindi': 300,
            'portuguese': 400,
            'spanish': 700,
            'french': 500,
            'german': 800,
            'italian': 500,
        }
    elif args.model_size == 3:
        lang_checkpoints = {
            'english': 400, 
            'hindi': 200,
            'portuguese': 300,
            'spanish': 400,
            'french': 400,
            'german': 500,
            'italian': 400,
        }
    elif args.model_size == 8:
        lang_checkpoints = {
            'english': 200, 
            'hindi': 300,
            'portuguese': 300,
            'spanish': 400,
            'french': 500,
            'german': 500,
            'italian': 500,
        }



    
    model_language = args.model_language
    language = args.language
    
    if args.finetuned:
        model = AutoModelForCausalLM.from_pretrained(f"../experiments/llama-{args.model_size}b/dropout-0.1-lr-5e-06/{model_language}/checkpoint-{lang_checkpoints[model_language]}").to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Llama-3.1-{args.model_size}B").to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-3.1-{args.model_size}B")
    tokenizer.pad_token = tokenizer.eos_token    


    # Formatting function
    def formatting_prompts_func(example):
        return {"prompt": [f"### Question: {q}\n ### Answer: {a}" for q, a in zip(example['question'], example['answer'])]}

    # Data collator
    response_template = " ### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


    if language == 'english':
        df_train = pd.read_csv(f'../data/{language}/train.csv')[['id', 'question', 'answer']].dropna()
        df_val = pd.read_csv(f'../data/{language}/dev.csv')[['id', 'question', 'answer']].dropna()
        df_test = pd.read_csv(f'../data/{language}/test.csv')[['id', 'question', 'answer']].dropna()
    else:
        df_train = pd.read_csv(f'../data/{language}/train.csv')[['id', 'question', 'translated_answer']]
        df_train = df_train.rename(columns={"translated_answer": "answer"})
        df_val = pd.read_csv(f'../data/{language}/dev.csv')[['id', 'question', 'translated_answer']]
        df_val = df_val.rename(columns={"translated_answer": "answer"})
        df_test = pd.read_csv(f'../data/{language}/test.csv')[['id', 'question', 'translated_answer']].dropna()
        df_test = df_test.rename(columns={"translated_answer": "answer"})

    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    if args.finetuned:
        output_dir = f"../results/outputs_{args.model_size}b/{model_language}-{language}"
    else:
        output_dir = f"../results/outputs_{args.model_size}b/pretrained-{language}"
    os.makedirs(output_dir, exist_ok=True)


    accuracy, df = calculate_accuracy(model, tokenizer, device, df_test)
    df.to_csv(f"{output_dir}/test.csv", index=False)
    # accuracy, df = calculate_accuracy(model, tokenizer, device, df_val)
    # df.to_csv(f"{output_dir}/val.csv", index=False)
    # accuracy, df = calculate_accuracy(model, tokenizer, device, df_train)
    # df.to_csv(f"{output_dir}/train.csv", index=False)