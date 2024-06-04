import optuna
import numpy as np
from transformers import TrainingArguments, Trainer, RobertaForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, DataCollatorWithPadding
from datasets import load_metric, Dataset
from optuna.integration.kfp import save_artifact
from sklearn.model_selection import train_test_split
import argparse
from huggingface_hub import HfFolder
import pandas as pd

def split_for_hyperparameter_tuning(train_roberta, validation_roberta):
    # TRAINING SPLIT
    X_train_full = train_roberta.drop(columns=['label'])
    y_train_full = train_roberta['label']

    # Perform stratified split on training data
    _, X_val, _, y_val = train_test_split(X_train_full, y_train_full, test_size=0.20, stratify=y_train_full, random_state=42)

    train_roberta_20_percent = X_val.copy()
    train_roberta_20_percent['label'] = y_val

    # VALIDATION SPLIT
    X_val_full = validation_roberta.drop(columns=['label'])
    y_val_full = validation_roberta['label']

    # Perform stratified split on validation data
    _, X_val, _, y_val = train_test_split(X_val_full, y_val_full, test_size=0.20, stratify=y_val_full, random_state=42)

    validation_roberta_20_percent = X_val.copy()
    validation_roberta_20_percent['label'] = y_val

    return train_roberta_20_percent, validation_roberta_20_percent

def tokenize_function(example):
    return tokenizer(example["text"], max_length=512, padding=True, truncation=True)

def objective(trial):
    # Suggest values for the hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 4e-5, 6e-5)
    weight_decay = trial.suggest_uniform('weight_decay', 0.008, 0.03)
    num_train_epochs = trial.suggest_int('num_train_epochs', 4, 10)

    # Define the compute_metrics function
    def compute_metrics(eval_preds):
        metric = load_metric("glue", "mrpc")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=repository_id,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="epoch",
        logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
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

    # Initialize EarlyStoppingCallback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,  # Number of evaluation steps with no improvement
        early_stopping_threshold=0.0  # Minimum improvement to be considered
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_for_hyperparameter_tuning_dataset,
        eval_dataset=tokenized_test_for_hyperparameter_tuning_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]  # Add early stopping callback
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_result = trainer.evaluate()
    return eval_result['eval_loss']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository_id', type=str, required=True)
    parser.add_argument('--train_dataset', type=str, required=True)
    parser.add_argument('--test_dataset', type=str, required=True)
    args = parser.parse_args()

    global repository_id, model_id, tokenized_train_for_hyperparameter_tuning_dataset
    global tokenized_test_for_hyperparameter_tuning_dataset, data_collator, tokenizer

    repository_id = args.repository_id
    model_id = "roberta-base"

    # Load datasets
    train_dataset = pd.read_csv(args.train_dataset)
    test_dataset = pd.read_csv(args.test_dataset)

    train_for_hyperparameter_tuning, validation_for_hyperparameter_tuning = split_for_hyperparameter_tuning(train_dataset, test_dataset)

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = Dataset.from_pandas(train_for_hyperparameter_tuning)
    validation_dataset = Dataset.from_pandas(validation_for_hyperparameter_tuning)

    tokenized_train_for_hyperparameter_tuning_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_for_hyperparameter_tuning_dataset = validation_dataset.map(tokenize_function, batched=True)

    # Create Optuna study and optimize
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # Save the best trial
    best_trial = study.best_trial
    save_artifact(best_trial, 'best_trial.json')

    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
