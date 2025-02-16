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
    
lang_checkpoints = {
    'english': 300, 
    'hindi': 300,
    'portuguese': 400,
    'spanish': 700,
    'french': 500,
    'german': 800,
    'italian': 500,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--starting_language', type=str, default='english')
    parser.add_argument('--finetuned_language', type=str, default='french')
    parser.add_argument('--unfreezed_module', type=str, default='full')
    parser.add_argument('--model_type', type=str, default='llama-1b')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == 'llama-1b':

        partialy_finetuned_model_path = f"../experiments/llama-1b/partial-{0}-{5}-{args.unfreezed_module}/{args.starting_language}-{args.finetuned_language}"
        # partialy_finetuned_model_path = f"../experiments/llama-1b/freezed-0-5/french/checkpoint-200"
        partialy_finetuned_model_path = os.path.join(partialy_finetuned_model_path, os.listdir(partialy_finetuned_model_path)[0])
        partialy_finetuned_model_0_5 = AutoModelForCausalLM.from_pretrained(partialy_finetuned_model_path).to(device)
        partialy_finetuned_model_0_5.eval()

        partialy_finetuned_model_path = f"../experiments/llama-1b/partial-{0}-{10}-{args.unfreezed_module}/{args.starting_language}-{args.finetuned_language}"
        # partialy_finetuned_model_path = f"../experiments/llama-1b/freezed-0-10/french/checkpoint-200"
        partialy_finetuned_model_path = os.path.join(partialy_finetuned_model_path, os.listdir(partialy_finetuned_model_path)[0])
        partialy_finetuned_model_0_10 = AutoModelForCausalLM.from_pretrained(partialy_finetuned_model_path).to(device)
        partialy_finetuned_model_0_10.eval()

        partialy_finetuned_model_path = f"../experiments/llama-1b/partial-{10}-{16}-{args.unfreezed_module}/{args.starting_language}-{args.finetuned_language}"
        # partialy_finetuned_model_path = f"../experiments/llama-1b/freezed-10-16/french/checkpoint-200"
        partialy_finetuned_model_path = os.path.join(partialy_finetuned_model_path, os.listdir(partialy_finetuned_model_path)[0])
        partialy_finetuned_model_10_16 = AutoModelForCausalLM.from_pretrained(partialy_finetuned_model_path).to(device)
        partialy_finetuned_model_10_16.eval()

        finetuned_model_path = f'../experiments/llama-1b/dropout-0.1-lr-5e-06/{args.finetuned_language}/checkpoint-{lang_checkpoints[args.finetuned_language]}'
        finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path).to(device)
        finetuned_model.eval()

        pretrained_model_path = f'../experiments/llama-1b/dropout-0.1-lr-5e-06/{args.starting_language}/checkpoint-{lang_checkpoints[args.starting_language]}'
        pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_path).to(device)
        pretrained_model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
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

        pretrained_model_accuracy = {
            # f"val_{args.finetuned_language}": calculate_accuracy(pretrained_model, tokenizer, val_unswap_dataloader, device),
            f"test_{args.finetuned_language}": calculate_accuracy(pretrained_model, tokenizer, test_unswap_dataloader, device),
            # f"val_{args.starting_language}": calculate_accuracy(pretrained_model, tokenizer, val_swap_dataloader, device),
            f"test_{args.starting_language}": calculate_accuracy(pretrained_model, tokenizer, test_swap_dataloader, device)
        }

        finetuned_model_accuracy = {
            # f"val_{args.finetuned_language}": calculate_accuracy(finetuned_model, tokenizer, val_unswap_dataloader, device),
            f"test_{args.finetuned_language}": calculate_accuracy(finetuned_model, tokenizer, test_unswap_dataloader, device),
            # f"val_{args.starting_language}": calculate_accuracy(finetuned_model, tokenizer, val_swap_dataloader, device),
            f"test_{args.starting_language}": calculate_accuracy(finetuned_model, tokenizer, test_swap_dataloader, device)
        }

        # print(f"Pretrained model accuracy: {pretrained_model_accuracy}")
        # print(f"Finetuned model accuracy: {finetuned_model_accuracy}")

        partialy_finetuned_model_0_5_accuracy = {
            # f"val_{args.finetuned_language}": calculate_accuracy(partialy_finetuned_model_0_5, tokenizer, val_unswap_dataloader, device),
            f"test_{args.finetuned_language}": calculate_accuracy(partialy_finetuned_model_0_5, tokenizer, test_unswap_dataloader, device),
            # f"val_{args.starting_language}": calculate_accuracy(partialy_finetuned_model_0_5, tokenizer, val_swap_dataloader, device),
            f"test_{args.starting_language}": calculate_accuracy(partialy_finetuned_model_0_5, tokenizer, test_swap_dataloader, device)
        }

        partialy_finetuned_model_0_10_accuracy = {
            # f"val_{args.finetuned_language}": calculate_accuracy(partialy_finetuned_model_0_10, tokenizer, val_unswap_dataloader, device),
            f"test_{args.finetuned_language}": calculate_accuracy(partialy_finetuned_model_0_10, tokenizer, test_unswap_dataloader, device),
            # f"val_{args.starting_language}": calculate_accuracy(partialy_finetuned_model_0_10, tokenizer, val_swap_dataloader, device),
            f"test_{args.starting_language}": calculate_accuracy(partialy_finetuned_model_0_10, tokenizer, test_swap_dataloader, device)
        }

        partialy_finetuned_model_10_16_accuracy = { 
            # f"val_{args.finetuned_language}": calculate_accuracy(partialy_finetuned_model_10_16, tokenizer, val_unswap_dataloader, device),
            f"test_{args.finetuned_language}": calculate_accuracy(partialy_finetuned_model_10_16, tokenizer, test_unswap_dataloader, device),
            # f"val_{args.starting_language}": calculate_accuracy(partialy_finetuned_model_10_16, tokenizer, val_swap_dataloader, device),
            f"test_{args.starting_language}": calculate_accuracy(partialy_finetuned_model_10_16, tokenizer, test_swap_dataloader, device)
        }


        df = pd.DataFrame([pretrained_model_accuracy, partialy_finetuned_model_0_5_accuracy, partialy_finetuned_model_0_10_accuracy, partialy_finetuned_model_10_16_accuracy, finetuned_model_accuracy])
        df['model'] = [f'fully-{args.starting_language}-trained', f'partialy_{args.finetuned_language}-tuned_0_5', f'partialy_{args.finetuned_language}-tuned_0_10', f'partialy_{args.finetuned_language}-tuned_10_16', f'fully-{args.finetuned_language}-trained']
        df = df.melt(id_vars='model', var_name='split', value_name='accuracy')

        vals = {}
        val_keys = [f'fully-{args.starting_language}-trained', f'partialy_{args.finetuned_language}-tuned_0_5', f'partialy_{args.finetuned_language}-tuned_0_10', f'partialy_{args.finetuned_language}-tuned_10_16', f'fully-{args.finetuned_language}-trained']

        for key in val_keys:
            vals[key] = {}
            vals[key]['accuracy'] = df[(df['model'] == key) & (df['split'] != 'test_english')]['accuracy'].values[0] + df[(df['model'] == key) & (df['split'] == 'test_english')]['accuracy'].values[0]
            vals[key]['ratio'] = df[(df['model'] == key) & (df['split'] != 'test_english')]['accuracy'].values[0] 

        fig, ax = plt.subplots()
        fig.set_size_inches(10, 6)
        accuracy_bars = ax.bar(vals.keys(), [vals[key]['accuracy'] for key in vals.keys()], color='aliceblue', edgecolor='xkcd:denim blue', label='Cum. Acc.', hatch='//', alpha=0.5)

        # Create ratio bars
        ratio_bars = ax.bar(vals.keys(), [vals[key]['ratio'] for key in vals.keys()], color='xkcd:denim blue', label=f'{args.finetuned_language} Acc.', alpha=0.9)

        # Rotate bar labels
        ax.set_xticklabels(vals.keys(), rotation=45, ha='right')

        plt.title(f"{args.finetuned_language} - Llama-1B")
        plt.legend()
        plt.savefig(f"../results/coca-cola-partial/llama-1b/{args.starting_language}-{args.finetuned_language}-{args.unfreezed_module}.png", dpi=300, bbox_inches='tight')

        # plt.figure(figsize=(12, 6))
        # sns.barplot(x='model', y='accuracy', hue='split', data=df)
        # plt.title(f"Accuracy of models on {args.finetuned_language} inputs")
        # plt.show()
        # # save the plot
        # plt.savefig(f"../results/llama-1b-partial/loss_based_2/{args.starting_language}-{args.finetuned_language}-{args.unfreezed_module}.png", dpi=300)
        # # save the dataframe
        df.to_csv(f"../results/coca-cola-partial/llama-1b/{args.starting_language}-{args.finetuned_language}-{args.unfreezed_module}.csv", index=False)

    if args.model_type == 'llama-3b':


        lang_checkpoints = {
            'english': 400, 
            'hindi': 200,
            'portuguese': 300,
            'spanish': 400,
            'french': 400,
            'german': 500,
            'italian': 400,
        }

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
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



        partialy_finetuned_model_path = f"../experiments/llama-3b/partial-{0}-{14}-{args.unfreezed_module}/{args.starting_language}-{args.finetuned_language}"
        # partialy_finetuned_model_path = f"../experiments/llama-1b/freezed-0-5/french/checkpoint-200"
        partialy_finetuned_model_path = os.path.join(partialy_finetuned_model_path, os.listdir(partialy_finetuned_model_path)[0])
        partialy_finetuned_model_0_14 = AutoModelForCausalLM.from_pretrained(partialy_finetuned_model_path).to(device)
        partialy_finetuned_model_0_14.eval()

        partialy_finetuned_model_0_14_accuracy = {
            # f"val_{args.finetuned_language}": calculate_accuracy(partialy_finetuned_model_0_5, tokenizer, val_unswap_dataloader, device),
            f"test_{args.finetuned_language}": calculate_accuracy(partialy_finetuned_model_0_14, tokenizer, test_unswap_dataloader, device),
            # f"val_{args.starting_language}": calculate_accuracy(partialy_finetuned_model_0_14, tokenizer, val_swap_dataloader, device),
            f"test_{args.starting_language}": calculate_accuracy(partialy_finetuned_model_0_14, tokenizer, test_swap_dataloader, device)
        }

        del partialy_finetuned_model_0_14

        partialy_finetuned_model_path = f"../experiments/llama-3b/partial-{14}-{27}-{args.unfreezed_module}/{args.starting_language}-{args.finetuned_language}"
        # partialy_finetuned_model_path = f"../experiments/llama-1b/freezed-0-10/french/checkpoint-200"
        partialy_finetuned_model_path = os.path.join(partialy_finetuned_model_path, os.listdir(partialy_finetuned_model_path)[0])
        partialy_finetuned_model_14_27 = AutoModelForCausalLM.from_pretrained(partialy_finetuned_model_path).to(device)
        partialy_finetuned_model_14_27.eval()

        partialy_finetuned_model_14_27_accuracy = {
            # f"val_{args.finetuned_language}": calculate_accuracy(partialy_finetuned_model_0_10, tokenizer, val_unswap_dataloader, device),
            f"test_{args.finetuned_language}": calculate_accuracy(partialy_finetuned_model_14_27, tokenizer, test_unswap_dataloader, device),
            # f"val_{args.starting_language}": calculate_accuracy(partialy_finetuned_model_14_27, tokenizer, val_swap_dataloader, device),
            f"test_{args.starting_language}": calculate_accuracy(partialy_finetuned_model_14_27, tokenizer, test_swap_dataloader, device)
        }

        del partialy_finetuned_model_14_27


        finetuned_model_path = f'../experiments/llama-3b/dropout-0.1-lr-5e-06/{args.finetuned_language}/checkpoint-{lang_checkpoints[args.finetuned_language]}'
        finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path).to(device)
        finetuned_model.eval()

        finetuned_model_accuracy = {
            # f"val_{args.finetuned_language}": calculate_accuracy(finetuned_model, tokenizer, val_unswap_dataloader, device),
            f"test_{args.finetuned_language}": calculate_accuracy(finetuned_model, tokenizer, test_unswap_dataloader, device),
            # f"val_{args.starting_language}": calculate_accuracy(finetuned_model, tokenizer, val_swap_dataloader, device),
            f"test_{args.starting_language}": calculate_accuracy(finetuned_model, tokenizer, test_swap_dataloader, device)
        }

        del finetuned_model

        pretrained_model_path = f'../experiments/llama-3b/dropout-0.1-lr-5e-06/{args.starting_language}/checkpoint-{lang_checkpoints[args.starting_language]}'
        pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_path).to(device)
        pretrained_model.eval()

        pretrained_model_accuracy = {
            # f"val_{args.finetuned_language}": calculate_accuracy(pretrained_model, tokenizer, val_unswap_dataloader, device),
            f"test_{args.finetuned_language}": calculate_accuracy(pretrained_model, tokenizer, test_unswap_dataloader, device),
            # f"val_{args.starting_language}": calculate_accuracy(pretrained_model, tokenizer, val_swap_dataloader, device),
            f"test_{args.starting_language}": calculate_accuracy(pretrained_model, tokenizer, test_swap_dataloader, device)
        }
        
        


        df = pd.DataFrame([pretrained_model_accuracy, partialy_finetuned_model_0_14_accuracy, partialy_finetuned_model_14_27_accuracy, finetuned_model_accuracy])
        df['model'] = [f'fully-{args.starting_language}-trained', f'partialy_{args.finetuned_language}-tuned_0_14', f'partialy_{args.finetuned_language}-tuned_14_27', f'fully-{args.finetuned_language}-trained']
        df = df.melt(id_vars='model', var_name='split', value_name='accuracy')

        vals = {}
        val_keys = [f'fully-{args.starting_language}-trained', f'partialy_{args.finetuned_language}-tuned_0_14', f'partialy_{args.finetuned_language}-tuned_14_27', f'fully-{args.finetuned_language}-trained']

        for key in val_keys:
            vals[key] = {}
            vals[key]['accuracy'] = df[(df['model'] == key) & (df['split'] != 'test_english')]['accuracy'].values[0] + df[(df['model'] == key) & (df['split'] == 'test_english')]['accuracy'].values[0]
            vals[key]['ratio'] = df[(df['model'] == key) & (df['split'] != 'test_english')]['accuracy'].values[0] 

        fig, ax = plt.subplots()
        fig.set_size_inches(10, 6)
        accuracy_bars = ax.bar(vals.keys(), [vals[key]['accuracy'] for key in vals.keys()], color='aliceblue', edgecolor='xkcd:denim blue', label='Cum. Acc.', hatch='//', alpha=0.5)

        # Create ratio bars
        ratio_bars = ax.bar(vals.keys(), [vals[key]['ratio'] for key in vals.keys()], color='xkcd:denim blue', label=f'{args.finetuned_language} Acc.', alpha=0.9)

        # Rotate bar labels
        ax.set_xticklabels(vals.keys(), rotation=45, ha='right')
        plt.title(f"{args.finetuned_language} - Llama-3B")

        plt.legend()
        plt.savefig(f"../results/coca-cola-partial/llama-3b/{args.starting_language}-{args.finetuned_language}-{args.unfreezed_module}.png", dpi=300, bbox_inches='tight')

        # plt.figure(figsize=(12, 6))
        # sns.barplot(x='model', y='accuracy', hue='split', data=df)
        # plt.title(f"Accuracy of models on {args.finetuned_language} inputs")
        # plt.show()
        # # save the plot
        # plt.savefig(f"../results/llama-1b-partial/loss_based_2/{args.starting_language}-{args.finetuned_language}-{args.unfreezed_module}.png", dpi=300)
        # # save the dataframe
        df.to_csv(f"../results/coca-cola-partial/llama-3b/{args.starting_language}-{args.finetuned_language}-{args.unfreezed_module}.csv", index=False)


    if args.model_type == 'llama-8b':


        lang_checkpoints = {
            'english': 200, 
            'hindi': 300,
            'portuguese': 300,
            'spanish': 400,
            'french': 500,
            'german': 500,
            'italian': 500,
        }

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
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



        partialy_finetuned_model_path = f"../experiments/llama-8b/partial-{0}-{15}-{args.unfreezed_module}/{args.starting_language}-{args.finetuned_language}"
        # partialy_finetuned_model_path = f"../experiments/llama-1b/freezed-0-5/french/checkpoint-200"
        partialy_finetuned_model_path = os.path.join(partialy_finetuned_model_path, os.listdir(partialy_finetuned_model_path)[0])
        partialy_finetuned_model_0_14 = AutoModelForCausalLM.from_pretrained(partialy_finetuned_model_path).to(device)
        partialy_finetuned_model_0_14.eval()

        partialy_finetuned_model_0_14_accuracy = {
            # f"val_{args.finetuned_language}": calculate_accuracy(partialy_finetuned_model_0_5, tokenizer, val_unswap_dataloader, device),
            f"test_{args.finetuned_language}": calculate_accuracy(partialy_finetuned_model_0_14, tokenizer, test_unswap_dataloader, device),
            # f"val_{args.starting_language}": calculate_accuracy(partialy_finetuned_model_0_14, tokenizer, val_swap_dataloader, device),
            f"test_{args.starting_language}": calculate_accuracy(partialy_finetuned_model_0_14, tokenizer, test_swap_dataloader, device)
        }

        del partialy_finetuned_model_0_14

        partialy_finetuned_model_path = f"../experiments/llama-8b/partial-{15}-{31}-{args.unfreezed_module}/{args.starting_language}-{args.finetuned_language}"
        # partialy_finetuned_model_path = f"../experiments/llama-1b/freezed-0-10/french/checkpoint-200"
        partialy_finetuned_model_path = os.path.join(partialy_finetuned_model_path, os.listdir(partialy_finetuned_model_path)[0])
        partialy_finetuned_model_14_27 = AutoModelForCausalLM.from_pretrained(partialy_finetuned_model_path).to(device)
        partialy_finetuned_model_14_27.eval()

        partialy_finetuned_model_14_27_accuracy = {
            # f"val_{args.finetuned_language}": calculate_accuracy(partialy_finetuned_model_0_10, tokenizer, val_unswap_dataloader, device),
            f"test_{args.finetuned_language}": calculate_accuracy(partialy_finetuned_model_14_27, tokenizer, test_unswap_dataloader, device),
            # f"val_{args.starting_language}": calculate_accuracy(partialy_finetuned_model_14_27, tokenizer, val_swap_dataloader, device),
            f"test_{args.starting_language}": calculate_accuracy(partialy_finetuned_model_14_27, tokenizer, test_swap_dataloader, device)
        }

        del partialy_finetuned_model_14_27


        finetuned_model_path = f'../experiments/llama-8b/dropout-0.1-lr-5e-06/{args.finetuned_language}/checkpoint-{lang_checkpoints[args.finetuned_language]}'
        finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path).to(device)
        finetuned_model.eval()

        finetuned_model_accuracy = {
            # f"val_{args.finetuned_language}": calculate_accuracy(finetuned_model, tokenizer, val_unswap_dataloader, device),
            f"test_{args.finetuned_language}": calculate_accuracy(finetuned_model, tokenizer, test_unswap_dataloader, device),
            # f"val_{args.starting_language}": calculate_accuracy(finetuned_model, tokenizer, val_swap_dataloader, device),
            f"test_{args.starting_language}": calculate_accuracy(finetuned_model, tokenizer, test_swap_dataloader, device)
        }

        del finetuned_model

        pretrained_model_path = f'../experiments/llama-8b/dropout-0.1-lr-5e-06/{args.starting_language}/checkpoint-{lang_checkpoints[args.starting_language]}'
        pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_path).to(device)
        pretrained_model.eval()

        pretrained_model_accuracy = {
            # f"val_{args.finetuned_language}": calculate_accuracy(pretrained_model, tokenizer, val_unswap_dataloader, device),
            f"test_{args.finetuned_language}": calculate_accuracy(pretrained_model, tokenizer, test_unswap_dataloader, device),
            # f"val_{args.starting_language}": calculate_accuracy(pretrained_model, tokenizer, val_swap_dataloader, device),
            f"test_{args.starting_language}": calculate_accuracy(pretrained_model, tokenizer, test_swap_dataloader, device)
        }
        
        


        df = pd.DataFrame([pretrained_model_accuracy, partialy_finetuned_model_0_14_accuracy, partialy_finetuned_model_14_27_accuracy, finetuned_model_accuracy])
        df['model'] = [f'fully-{args.starting_language}-trained', f'partialy_{args.finetuned_language}-tuned_0_15', f'partialy_{args.finetuned_language}-tuned_15_31', f'fully-{args.finetuned_language}-trained']
        df = df.melt(id_vars='model', var_name='split', value_name='accuracy')

        vals = {}
        val_keys = [f'fully-{args.starting_language}-trained', f'partialy_{args.finetuned_language}-tuned_0_15', f'partialy_{args.finetuned_language}-tuned_15_31', f'fully-{args.finetuned_language}-trained']

        for key in val_keys:
            vals[key] = {}
            vals[key]['accuracy'] = df[(df['model'] == key) & (df['split'] != 'test_english')]['accuracy'].values[0] + df[(df['model'] == key) & (df['split'] == 'test_english')]['accuracy'].values[0]
            vals[key]['ratio'] = df[(df['model'] == key) & (df['split'] != 'test_english')]['accuracy'].values[0] 

        fig, ax = plt.subplots()
        fig.set_size_inches(10, 6)
        accuracy_bars = ax.bar(vals.keys(), [vals[key]['accuracy'] for key in vals.keys()], color='aliceblue', edgecolor='xkcd:denim blue', label='Cum. Acc.', hatch='//', alpha=0.5)

        # Create ratio bars
        ratio_bars = ax.bar(vals.keys(), [vals[key]['ratio'] for key in vals.keys()], color='xkcd:denim blue', label=f'{args.finetuned_language} Acc.', alpha=0.9)

        # Rotate bar labels
        ax.set_xticklabels(vals.keys(), rotation=45, ha='right')
        plt.title(f"{args.finetuned_language} - Llama-8B")

        plt.legend()
        plt.savefig(f"../results/coca-cola-partial/llama-8b/{args.starting_language}-{args.finetuned_language}-{args.unfreezed_module}.png", dpi=300, bbox_inches='tight')

        # plt.figure(figsize=(12, 6))
        # sns.barplot(x='model', y='accuracy', hue='split', data=df)
        # plt.title(f"Accuracy of models on {args.finetuned_language} inputs")
        # plt.show()
        # # save the plot
        # plt.savefig(f"../results/llama-1b-partial/loss_based_2/{args.starting_language}-{args.finetuned_language}-{args.unfreezed_module}.png", dpi=300)
        # # save the dataframe
        df.to_csv(f"../results/coca-cola-partial/llama-8b/{args.starting_language}-{args.finetuned_language}-{args.unfreezed_module}.csv", index=False)