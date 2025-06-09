# finetune the model

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from dataloader import TweetDataset
import argparse
from sklearn.model_selection import train_test_split
import os

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/flan-t5-large")
    parser.add_argument("--device", type=str, default="None")
    parser.add_argument("--data_file", type=str, default="data/Q2_20230202_majority.csv")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--prompt_type", type=str, default="baseline")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--wandb", action="store_true")

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # load the data
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    if args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q", "v", "k", "o"],
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # load the dataset
    df = pd.read_csv(args.data_file)
    # drop na and duplicates
    df = df.dropna(subset=['tweet', 'label_majority'])
    df = df.drop_duplicates(subset=['tweet'], keep='first')

    # split the data
    df_train, df_eval = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_majority'])
    dataset_train = TweetDataset(df_train, train=True, tokenizer=tokenizer, prompt_type=args.prompt_type)
    dataset_eval = TweetDataset(df_eval, train=True, tokenizer=tokenizer, prompt_type=args.prompt_type)

    print(f"Training dataset size: {len(dataset_train)}")
    print(f"Evaluation dataset size: {len(dataset_eval)}")

    # load the dataloader
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")

    # check dir exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(f"{args.output_dir}/logs")

    # training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        run_name="lora_finetune",
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=100,
        save_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="wandb" if args.wandb else None,  # Disable wandb logging
    )

    # load the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # train the model
    trainer.train()

    # save the model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()