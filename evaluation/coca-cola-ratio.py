from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def calculate_accuracy(model, tokenizer, dataloader, device):
    total, correct = 0, 0
    for batch in dataloader:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        labels = torch.where(labels == -100, torch.tensor(tokenizer.eos_token_id), labels)
        # Remove answer part from inputs
        inputs_no_answer = []
        for input_ids in inputs:
            input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            question_part = input_text.split(" ### Answer:")[0]
            question_part = question_part + " ### Answer: "
            # print(question_part)
            inputs_no_answer.append(tokenizer(question_part, return_tensors="pt").input_ids.squeeze(0))
        
        
        inputs_no_answer = torch.nn.utils.rnn.pad_sequence(inputs_no_answer, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        attention_mask_no_answer = (inputs_no_answer != tokenizer.pad_token_id).long().to(device)
        
        outputs = model.generate(inputs_no_answer, attention_mask=attention_mask_no_answer, do_sample=False, max_new_tokens=20)
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        references = tokenizer.batch_decode(labels, skip_special_tokens=True)
        for pred, ref in zip(predictions, references):
            if (pred.strip().lower() in ref.strip().lower()) or (ref.strip().lower() in pred.strip().lower()):
                correct += 1
            total += 1

    return correct / total

def load_data(language):
    if language == 'english':
        df_train = pd.read_csv(f'../data/{language}/train.csv')[['question', 'answer']]
        df_val = pd.read_csv(f'../data/{language}/dev.csv')[['question', 'answer']]
        df_test = pd.read_csv(f'../data/{language}/test.csv')[['question', 'answer']]
    else:
        df_train = pd.read_csv(f'../data/{language}/train.csv')[['question', 'translated_answer']]
        df_train = df_train.rename(columns={"translated_answer": "answer"})
        df_val = pd.read_csv(f'../data/{language}/dev.csv')[['question', 'translated_answer']]
        df_val = df_val.rename(columns={"translated_answer": "answer"})
        df_test = pd.read_csv(f'../data/{language}/test.csv')[['question', 'translated_answer']]
        df_test = df_test.rename(columns={"translated_answer": "answer"})
    return df_train, df_val, df_test

def filter_data(df_val, df, swap=False):
    df_val['answer'] = df_val['answer'].str.lower()
    df['answer'] = df['answer'].str.lower()
    diff = df_val[df_val['answer'] != df['answer']]
    df = df[df['answer'] != df_val['answer']]

    diff['en_answer'] = df['answer']
    diff.dropna()
    if swap:
        diff['answer'] = diff['en_answer']
    
    return diff[['question', 'answer']].dropna()


def get_data_loader(df, tokenizer, collator):
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=["question", "answer"])
    dataset = dataset.map(lambda ex: tokenizer(ex["prompt"], truncation=True), batched=True, remove_columns=["prompt"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, collate_fn=collator)
    return dataloader
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--starting_language', type=str, default='english')
    parser.add_argument('--finetuned_language', type=str, default='french')
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


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_size == 8:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-3.2-{args.model_size}B")

    tokenizer.pad_token = tokenizer.eos_token    


    # Formatting function
    def formatting_prompts_func(example):
        return {"prompt": [f"### Question: {q}\n ### Answer: {a}" for q, a in zip(example['question'], example['answer'])]}

    # Data collator
    response_template = " ### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


    _, df_val, df_test = load_data(args.finetuned_language)
    _, df_val_starting_language, df_test_starting_language = load_data(args.starting_language)
    
    df_val_swap = filter_data(df_val, df_val_starting_language, swap=True)
    df_test_swap = filter_data(df_test, df_test_starting_language, swap=True)
    df_val_unswap = filter_data(df_val, df_val_starting_language)
    df_test_unswap = filter_data(df_test, df_test_starting_language)

    val_swap_dataloader = get_data_loader(df_val_swap, tokenizer, collator)
    test_swap_dataloader = get_data_loader(df_test_swap, tokenizer, collator)
    val_unswap_dataloader = get_data_loader(df_val_unswap, tokenizer, collator)
    test_unswap_dataloader = get_data_loader(df_test_unswap, tokenizer, collator)

    finetuned_model_path = f"../experiments/llama-{args.model_size}b/dropout-0.1-lr-5e-06/{args.finetuned_language}/checkpoint-{lang_checkpoints[args.finetuned_language]}"
    finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path).to(device)
    finetuned_model.eval()
    
    finetuned_model_accuracy = {
        # f"val_{args.finetuned_language}": calculate_accuracy(finetuned_model, tokenizer, val_unswap_dataloader, device),
        f"test_{args.finetuned_language}": calculate_accuracy(finetuned_model, tokenizer, test_unswap_dataloader, device),
        # f"val_{args.starting_language}": calculate_accuracy(finetuned_model, tokenizer, val_swap_dataloader, device),
        f"test_{args.starting_language}": calculate_accuracy(finetuned_model, tokenizer, test_swap_dataloader, device)
    }
    
    del finetuned_model

    starting_model_path = f"../experiments/llama-{args.model_size}b/dropout-0.1-lr-5e-06/{args.starting_language}/checkpoint-{lang_checkpoints[args.starting_language]}"
    starting_model = AutoModelForCausalLM.from_pretrained(starting_model_path).to(device)
    starting_model.eval()
    
    starting_model_accuracy = {
        # f"val_{args.finetuned_language}": calculate_accuracy(starting_model, tokenizer, val_unswap_dataloader, device),
        f"test_{args.finetuned_language}": calculate_accuracy(starting_model, tokenizer, test_unswap_dataloader, device),
        # f"val_{args.starting_language}": calculate_accuracy(starting_model, tokenizer, val_swap_dataloader, device),
        f"test_{args.starting_language}": calculate_accuracy(starting_model, tokenizer, test_swap_dataloader, device)
    }
    
    del starting_model
    if args.model_size == 8:
        pretrained_model_path = f"meta-llama/Llama-3.1-8B"
    else:
        pretrained_model_path = f"meta-llama/Llama-3.2-{args.model_size}B"

    pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_path).to(device)
    pretrained_model.eval()
    
    pretrained_model_accuracy = {
        # f"val_{args.finetuned_language}": calculate_accuracy(pretrained_model, tokenizer, val_unswap_dataloader, device),
        f"test_{args.finetuned_language}": calculate_accuracy(pretrained_model, tokenizer, test_unswap_dataloader, device),
        # f"val_{args.starting_language}": calculate_accuracy(pretrained_model, tokenizer, val_swap_dataloader, device),
        f"test_{args.starting_language}": calculate_accuracy(pretrained_model, tokenizer, test_swap_dataloader, device)
    }


   
    df = pd.DataFrame([pretrained_model_accuracy, starting_model_accuracy, finetuned_model_accuracy])
    df['model'] = ["Pretrained", f"{args.starting_language}-tuned", "Finetuned"]
    df = df.melt(id_vars='model', var_name='split', value_name='accuracy')
    df.to_csv(f"../results/coca-cola-{args.model_size}b/{args.finetuned_language}.csv", index=False)
