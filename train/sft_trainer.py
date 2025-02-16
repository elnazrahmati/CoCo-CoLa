import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, AutoConfig
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import os
import wandb
import torch
import argparse
import numpy as np
from accelerate import PartialState



def calculate_accuracy(model, tokenizer, dataloader, device):
    total, correct = 0, 0
    for batch in dataloader:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        labels = torch.where(labels == -100, torch.tensor(tokenizer.eos_token_id), labels)
        inputs_no_answer = []
        for input_ids in inputs:
            input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            question_part = input_text.split(" ### Answer:")[0]
            inputs_no_answer.append(tokenizer(question_part, return_tensors="pt").input_ids.squeeze(0))
        
        inputs_no_answer = torch.nn.utils.rnn.pad_sequence(inputs_no_answer, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        attention_mask_no_answer = (inputs_no_answer != tokenizer.pad_token_id).long()
        
        outputs = model.generate(inputs_no_answer, attention_mask=attention_mask_no_answer, do_sample=False, max_new_tokens=10)
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        references = tokenizer.batch_decode(labels, skip_special_tokens=True)
        for pred, ref in zip(predictions, references):
            if (pred.strip().lower() in ref.strip().lower()) or (ref.strip().lower() in pred.strip().lower()):
                correct += 1
            total += 1

    return correct / total


if __name__ == '__main__':
    # add parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_language', type=str, default='french')
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')    
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--save_total_limit', type=int, default=1)


    args = parser.parse_args()

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="coca-cola")

    # set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    partial_state = PartialState()
    with partial_state.main_process_first():

        # Load and preprocess datasets
        if args.training_language == 'english':
            df_train = pd.read_csv(f'../data/{args.training_language}/train.csv')[['question', 'answer']].dropna()
            df_val = pd.read_csv(f'../data/{args.training_language}/dev.csv')[['question', 'answer']].dropna()
            df_test = pd.read_csv(f'../data/{args.training_language}/test.csv')[['question', 'answer']].dropna()
        else:
            df_train = pd.read_csv(f'../data/{args.training_language}/train.csv')[['question', 'translated_answer']].dropna()
            df_train = df_train.rename(columns={"translated_answer": "answer"})
            df_val = pd.read_csv(f'../data/{args.training_language}/dev.csv')[['question', 'translated_answer']].dropna()
            df_val = df_val.rename(columns={"translated_answer": "answer"})
            df_test = pd.read_csv(f'../data/{args.training_language}/test.csv')[['question', 'translated_answer']].dropna()
            df_test = df_test.rename(columns={"translated_answer": "answer"})
            
        train_dataset = Dataset.from_pandas(df_train)
        val_dataset = Dataset.from_pandas(df_val)
        test_dataset = Dataset.from_pandas(df_test)

        os.environ["WANDB_PROJECT"] = "coca-cola"

        # Ensure datasets are not empty
        assert len(train_dataset) > 0, "Training dataset is empty"
        assert len(val_dataset) > 0, "Validation dataset is empty"
        assert len(test_dataset) > 0, "Test dataset is empty"

        # Load model and tokenizer
        config = AutoConfig.from_pretrained(args.model, attention_probs_dropout_prob=args.dropout)
        MODEL_CONFIG = {
            "attn_implementation": "flash_attention_2",
            "torch_dtype": torch.bfloat16,
        }
        model = AutoModelForCausalLM.from_pretrained(args.model, config=config, **MODEL_CONFIG)
        model.train()
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.model_max_length = args.max_length
        tokenizer.pad_token = tokenizer.eos_token


    partial_state.wait_for_everyone()


    # Formatting function
    def formatting_prompts_func(example):
        return [f"### Question: {q}\n ### Answer: {a}" for q, a in zip(example['question'], example['answer'])]

    # Data collator
    response_template = " ### Answer:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template,
        tokenizer=tokenizer,
        )

    # Output directory
    model_size = args.model[-2]
    output_dir = f"../experiments/llama-{model_size}b/dropout-{args.dropout}-lr-{args.lr}/{args.training_language}"
    
    os.makedirs(output_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=output_dir, 
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=0,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=args.logging_steps,  
        logging_first_step=True, 
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        learning_rate=args.lr,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False, 
        bf16=True,
        bf16_full_eval = True
        )

    class LogAccuracyCallback(TrainerCallback):
        def __init__(self, tokenizer, train_dataloader):
            super().__init__()
            self.tokenizer = tokenizer
            self.train_dataloader = train_dataloader

        def on_evaluate(self, args, state, control, metrics, **kwargs):
            eval_dataloader = kwargs['eval_dataloader']
            # train_dataloader = self.train_dataloader
            model = kwargs['model']

            eval_accuracy = calculate_accuracy(model, self.tokenizer, eval_dataloader, args.device)
            # train_accuracy = calculate_accuracy(model, self.tokenizer, train_dataloader, args.device)
            
            # wandb.log({"train_accuracy": train_accuracy})
            wandb.log({"eval_accuracy": eval_accuracy})
            
            print(f"Val Accuracy: {eval_accuracy}")
            metrics.update({"eval_accuracy": eval_accuracy})
            # print(f"Train Accuracy: {train_accuracy}")


    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    train_dataloader = trainer.get_train_dataloader()
    log_accuracy_callback = LogAccuracyCallback(tokenizer, train_dataloader)
    trainer.add_callback(log_accuracy_callback)

    # Train
    # trainer.evaluate()
    trainer.train()