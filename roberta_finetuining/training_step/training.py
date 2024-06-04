import json
from transformers import TrainingArguments, Trainer, RobertaForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
import pandas as pd
from datasets import Dataset, load_metric
import numpy as np
from huggingface_hub import HfFolder
import argparse

# Define global variables
model_id = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def train_model(params_path, train_data_path, eval_data_path, repository_id):
    with open(params_path, 'r') as f:
        best_params = json.load(f)

    train_dataset = pd.read_csv(train_data_path)
    eval_dataset = pd.read_csv(eval_data_path)
    
    train = Dataset.from_pandas(train_dataset, preserve_index=False)
    validation = Dataset.from_pandas(eval_dataset, preserve_index=False)

    tokenized_train_dataset = train.map(tokenize_function, batched=True)
    tokenized_test_dataset = validation.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=repository_id,
        num_train_epochs=best_params['num_train_epochs'],
        evaluation_strategy="epoch",
        logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to="tensorboard",
        push_to_hub=True,
        hub_strategy="every_save",
        hub_model_id=repository_id,
        hub_token=HfFolder.get_token(),
    )

    model = RobertaForSequenceClassification.from_pretrained(model_id, num_labels=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(example):
    return tokenizer(example["text"], max_length=512, padding=True, truncation=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_path', type=str, required=True)
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--eval_data_path', type=str, required=True)
    parser.add_argument('--repository_id', type=str, required=True)
    args = parser.parse_args()

    train_model(args.params_path, args.train_data_path, args.eval_data_path, args.repository_id)

