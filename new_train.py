import dataclasses

import argparse
import os
import time

import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaConfig, RobertaTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from torch.utils.tensorboard import SummaryWriter
import evaluate
from datasets import load_dataset, load_from_disk

from helper import get_class, set_seed, get_tokenize_function, get_metric

import pandas as pd

@dataclasses.dataclass
class CustomTrainingArguments(TrainingArguments):
    model_name: str = ''
    dset: str = 'rte'
    class_num: int = 2
    num_input: int = 2
    filename: str = ''
    label_smoothing: float = 0.5

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        device = model.module.device if isinstance(model, nn.DataParallel) else model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # print(inputs.keys())

        # Forward pass with required arguments
        # outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        # logits = outputs

        # Custom loss calculation
        classifier_loss = self.loss(logits, inputs['labels'])

        # https://discuss.huggingface.co/t/evalprediction-has-an-unequal-number-of-label-ids-and-predictions/10557/4
        return (classifier_loss, {"label": logits}) if return_outputs else classifier_loss

def trainer(args):
    # Set training arguments
    training_args = CustomTrainingArguments(
        model_name = args.model_name,
        dset = args.dset,
        class_num = args.class_num,
        num_input = args.num_input,
        filename = args.filename,
        label_smoothing=0.1,

        output_dir=args.result_dir,
        eval_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.max_epoch,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
    )

    # # Load the tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(training_args.model_name)

    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if args.scratch:
        model_config = RobertaConfig(
            vocab_size=tokenizer.vocab_size,  # Specify the vocab size, same as tokenizer vocab size (commonly 30522 for BERT/RoBERTa)
            num_labels=training_args.class_num,  # Set the number of labels for sequence classification
            max_position_embeddings=514, # The maximum sequence length that this model might ever be used with
            num_attention_heads=12,      # Number of attention heads
            num_hidden_layers=12,        # Number of transformer encoder layers
            type_vocab_size=1,           # RoBERTa uses a type vocab size of 1
            hidden_size=768,             # Hidden size of the transformer layers
            intermediate_size=3072,      # Intermediate size in the feed-forward network
            hidden_act="gelu",           # Activation function used in hidden layers
            hidden_dropout_prob=0.1,     # Dropout probability for fully connected layers
            attention_probs_dropout_prob=0.1, # Dropout probability for attention probabilities
            layer_norm_eps=1e-5,         # Epsilon for layer normalization
            initializer_range=0.02,      # Standard deviation of the truncated_normal_initializer
        )
        # Create a new Roberta model with the above configuration
        model = RobertaForSequenceClassification(config=model_config)
    else:
        model = RobertaForSequenceClassification.from_pretrained(training_args.model_name, num_labels=training_args.class_num) #output_hidden_states=True

    # GPU
    # device = torch.device(f'cuda:0')

    device = torch.device('cuda')
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

        model.module.to(device)
    else:
        model.to(device)
    
    # Set up dataset
    test_size = 64

    if args.training_dataset == '':
        train_dataset = load_dataset("glue", training_args.dset, split='train').select(range(test_size))
        # print('RUNNING without training_dataset')
        # exit(0)
    else:
        train_dataset = load_from_disk(args.training_dataset).select(range(test_size))
        # print('RUNNING with training_dataset')
        # exit(0)

    tokenize_function = get_tokenize_function(training_args.num_input, tokenizer)
    tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)
    validation_dataset = load_from_disk(args.validation_dataset)#.select(range(test_size))
    tokenized_validation_datasets = validation_dataset.map(tokenize_function, batched=True)
    tokenized_train_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Load the metric for evaluation
    metric = evaluate.load("glue", training_args.dset)
    compute_metrics = get_metric(metric)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=training_args.logging_dir)

    # Log all hyperparameters using SummaryWriter for TensorBoard
    # training_args.to_dict() converts the hyperparameters to a dictionary
    for key, value in training_args.to_dict().items():
        writer.add_text(f"hyperparameters/{key}", str(value), 0)

    # Close the SummaryWriter after logging all the parameters
    writer.close()

    # Initialize the Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_datasets,
        eval_dataset=tokenized_validation_datasets,
        compute_metrics=compute_metrics,
        # data_collator=data_collator
    )

    # Train the model
    trainer.train()

    # model.save_pretrained(args.model_dir)
    model.module.save_pretrained(args.model_dir) if isinstance(model, nn.DataParallel) else model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.token_dir)

    tokenizer.save_pretrained(args.token_dir)

    # Evaluate the model
    # trainer.evaluate()
    # Evaluate the model on the full validation dataset after training
    final_eval_results = trainer.evaluate(eval_dataset=tokenized_validation_datasets)

    # Log the final evaluation metrics explicitly with a custom step or label
    trainer.log({
        "eval/final_accuracy": final_eval_results['eval_accuracy'],  # Example, log additional metrics as needed
        "eval/final_loss": final_eval_results['eval_loss']
    })

    # Add result to csv file
    # Define the path to the CSV file
    csv_file = "data.csv"

    # Check if the CSV file exists
    if os.path.exists(csv_file):
        # Load the existing CSV into a DataFrame
        df = pd.read_csv(csv_file)
        print("CSV loaded successfully.")
    else:
        # Define column names and data types
        column_types = {
            "Filename": str,
            "Epoch": int,            # Integer type for epoch numbers
            "Scratch": bool,        # Float type for scratch values
            "Finetune": str,    # String type for finetune data
            "Eval": str,         # String type for evaluation data
            "Seed": int
        }

        # Initialize DataFrame with column names and types
        df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in column_types.items()})
        print("CSV not found. Initialized a new DataFrame.")

    # Add new data (example data)
    if args.training_dataset == '':
        args.training_dataset = 'SAE'

    new_data = {"Filename": args.filename, "Epoch": args.max_epoch, "Scratch": args.scratch, "Finetune": args.training_dataset, "Eval": args.validation_dataset, "Seed": args.seed}
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

    # Save the DataFrame to the CSV file
    df.to_csv(csv_file, index=False)
    print(f"Data saved to {csv_file}.")

    # Optionally, you can print or save the final evaluation metrics
    print("Final Evaluation Results:", final_eval_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT adaptation for RoBERTa on GLUE tasks')
    parser.add_argument('--scratch', action='store_true', help='Train frmo scratch')

    parser.add_argument('--gpu_id', type=str, default='2', help="GPU id to use")
    parser.add_argument('--dset', type=str, default='rte', choices=["sst2", "mnli", "mrpc", "cola", "stsb", "qqp", "rte", "wnli", "qnli"], help="GLUE task name")
    parser.add_argument('--model_name', type=str, default='roberta-base', help="Pre-trained RoBERTa model")
    parser.add_argument('--logging_steps', type=int, default=10, help="Steps for logging")
    parser.add_argument('--save_total_limit', type=int, default=1, help="Limit the total number of checkpoints")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for initialization")

    parser.add_argument('--training_dataset', type=str, default='')
    parser.add_argument('--validation_dataset', type=str)

    parser.add_argument('--max_epoch', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for the optimizer")

    args = parser.parse_args()

    args.filename = args.model_name + '_' + args.dset + '_' + str(int(time.time()))

    # args.validation_dataset = f"./datasets/{args.dset}_validation_indian"
    args.logging_dir = f"./output/logs/train/{args.filename}"
    args.result_dir = f"./output/results/train/{args.filename}"
    args.model_dir = f"./output/models/train/{args.filename}_model"
    args.token_dir = f"./output/models/train/{args.filename}_tokenizer"

    for path in [args.logging_dir, args.result_dir, args.model_dir, args.token_dir]:
        if os.path.isfile(path):
            raise Exception(f"File already exists at {path}") 

    set_seed(args.seed)
    args.num_input, args.class_num = get_class(args.dset)

    # Set environment variables and seeds
    os.environ["CUDA_VISIBLE_DEVICES"] = "2" # args.gpu_id

    # Looping over the target domains to perform adaptation for each:        
    trainer(args)

# Baseline: output/models/train/roberta-base_rte_1730199414_model
# Skyline
