# Code is referenced from https://github.com/tim-learn/SHOT
# MIT License

# Copyright (c) 2020

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import os
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaModel, RobertaTokenizer, Trainer, TrainingArguments
import evaluate
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator

from helper import get_class, set_seed, BiGRU, Classifier, MultiModelWrapper

DATASET_PATH = ''

class FinetuneTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = {k: v for k, v in inputs.items()}

        # Forward pass
        outputs = model(inputs)
        
        # Custom loss calculation
        classifier_loss = self.loss(outputs, inputs['labels'])

        # https://discuss.huggingface.co/t/evalprediction-has-an-unequal-number-of-label-ids-and-predictions/10557/4
        return (classifier_loss, {"label": outputs}) if return_outputs else classifier_loss

def trainer(args, accelerator):
    device = torch.device('cuda')

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=args.result_dir,
        eval_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        label_names=["labels"],
        fp16=True,
        num_train_epochs=args.max_epoch,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
    )

    if accelerator.is_main_process:
        # Initialize TensorBoard SummaryWriter
        writer = SummaryWriter(log_dir=training_args.logging_dir)

        # Log all hyperparameters using SummaryWriter for TensorBoard
        # training_args.to_dict() converts the hyperparameters to a dictionary
        for key, value in training_args.to_dict().items():
            writer.add_text(f"hyperparameters/{key}", str(value), 0)

        for key, value in vars(args).items():
            writer.add_text(f"hyperparameters/{key}", str(value), 0)

        # Close the SummaryWriter after logging all the parameters
        writer.close()

    # # Load the tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)    
    netF = RobertaModel.from_pretrained(args.model_name).to(device)
    mid_hidden_size = netF.config.hidden_size // 2 # 256

    netB = BiGRU(input_size=netF.config.hidden_size, hidden_size=mid_hidden_size).to(device)
    netC = Classifier(hidden_size=mid_hidden_size, output_size=args.class_num).to(device)

    model = MultiModelWrapper(netF, netB, netC)

    # Freeze netF
    for param in model.netF.parameters():
        param.requires_grad = False

    if args.dset == 'besstie':
        # # BESSTIE Dataset
        train_dataset = load_dataset("json", data_files=f"{DATASET_PATH}/BESSTIE-sentiment/train/{args.training_dataset}.jsonl")['train']
        train_dataset = train_dataset.rename_column("text", "sentence")
        train_dataset = train_dataset.rename_column("sentiment_label", "label")

        validation_dataset = load_dataset("json", data_files=f"{DATASET_PATH}/BESSTIE-sentiment/valid/{args.validation_dataset}.jsonl")['train']
        validation_dataset = validation_dataset.rename_column("text", "sentence")
        validation_dataset = validation_dataset.rename_column("sentiment_label", "label")
        validation_dataset = validation_dataset.remove_columns('id')
    
    elif args.dset == 'amazon_text':
        # # AMAZON_TEXT
        train_dataset = load_dataset('json', data_files=f'{DATASET_PATH}/amazon_text/{args.training_dataset}.json', split='train')
        train_dataset = train_dataset.rename_column("review", "sentence")
        train_dataset = train_dataset.remove_columns(["summary", "helpful"])

        validation_dataset = load_dataset('json', data_files=f'{DATASET_PATH}/amazon_text/{args.validation_dataset}.json', split='train')
        validation_dataset = validation_dataset.rename_column("review", "sentence")
        validation_dataset = validation_dataset.remove_columns(["summary", "helpful"])

    elif args.dset in ['cola', 'mnli', 'qnli', 'rte', 'qqp', 'sst2', 'sts-b']:
        if args.training_dataset.endswith('_SAE_train'):
            train_dataset = load_dataset("glue", args.dset, split='train')
        else:
            train_dataset = load_from_disk(f'{DATASET_PATH}/multivalue/{args.training_dataset}')

        if args.validation_dataset.endswith('_SAE_validation'):
            validation_dataset = load_dataset("glue", args.dset, split='validation')
        else:
            validation_dataset = load_from_disk(f'{DATASET_PATH}/multivalue/{args.validation_dataset}')

    if args.train_size > 0:
        train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.train_size))

    # if args.val_size > 0:
    #     validation_dataset = validation_dataset.shuffle(seed=args.seed).select(range(args.val_size))
            
    def tokenize_function(examples):
        if args.dset == "rte":
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)
        elif args.dset == "qnli":
            return tokenizer(examples["question"], examples["sentence"], padding="max_length", truncation=True)
        else:
            return tokenizer(examples["sentence"], padding="max_length", truncation=True)

    tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)
    tokenized_train_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    tokenized_validation_datasets = validation_dataset.map(tokenize_function, batched=True)
    tokenized_validation_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Load the metric for evaluation
    if args.dset in ['cola', 'mnli', 'qnli', 'rte', 'qqp', 'sst2', 'sts-b']:
        metric = evaluate.load('glue', args.dset)
    else:
        metric = evaluate.load('accuracy')
    # compute_metrics = get_metric(metric)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        out = metric.compute(predictions=predictions, references=labels)
        return out
    
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    
    num_training_steps = len(tokenized_train_datasets) // training_args.per_device_train_batch_size * training_args.num_train_epochs

    # Custom learning rate lambda function
    def lr_lambda(current_step):
        p = current_step / num_training_steps
        return (1 + 10 * p) ** -0.75

    # Add param_group to the optimizer & use custom learning rate scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step))
    
    # Initialize the Trainer
    trainer = FinetuneTrainer(
        model = model,
        args=training_args,
        train_dataset=tokenized_train_datasets,
        eval_dataset=tokenized_validation_datasets,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
    )

    # Train the model
    trainer.train()

    torch.save(model.netB.state_dict(), args.netB_dir + '.pth')
    torch.save(model.netC.state_dict(), args.netC_dir + '.pth')
    
    # Evaluate the model
    final_eval_results = trainer.evaluate(eval_dataset=tokenized_validation_datasets)

    # Log the final evaluation metrics explicitly with a custom step or label
    
    if args.dset == 'cola':
        metric_name = 'eval_matthews_correlation'
    elif args.dset == 'sts-b':
        metric_name = 'eval_pearson'
    else:
        metric_name = 'eval_accuracy'
        
    # RTE QNLI MNLI QQP
    # SHOT+
        
    trainer.log({
        "eval/final_accuracy": final_eval_results[metric_name],  # Example, log additional metrics as needed
        "eval/final_loss": final_eval_results['eval_loss']
    })

    # Optionally, you can print or save the final evaluation metrics
    print("Final Evaluation Results:", final_eval_results)

    if accelerator.is_main_process:
        # Add result to csv file
        # Define the path to the CSV file
        csv_file = f"train.csv"

        # Check if the CSV file exists
        if os.path.exists(csv_file):
            # Load the existing CSV into a DataFrame
            df = pd.read_csv(csv_file)
            print("CSV loaded successfully.")
        else:
            # Define column names and data types
            column_types = {
                "Filename": str,
                "Dataset": str,
                "Epoch": int,            # Integer type for epoch numbers
                "Finetune": str,    # String type for finetune data
                "Eval": str,         # String type for evaluation data
                "Seed": int,
                "Acc": float,
                "Train_size": int,
                # "Val_size": int
            }

            # Initialize DataFrame with column names and types
            df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in column_types.items()})
            print("CSV not found. Initialized a new DataFrame.")

        new_data = {
            "Filename": args.filename, 
            "Dataset": args.dset, 
            "Epoch": args.max_epoch, 
            "Finetune": args.training_dataset, 
            "Eval": args.validation_dataset, 
            "Seed": args.seed, 
            "Acc": final_eval_results['eval_accuracy'], 
            "Train_size": args.train_size, 
            # "Val_size": args.val_size
        }
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

        # Save the DataFrame to the CSV file
        df.to_csv(csv_file, index=False)
        print(f"Data saved to {csv_file}.")

if __name__ == "__main__":
    accelerator = Accelerator()

    parser = argparse.ArgumentParser(description='SHOT adaptation for RoBERTa on GLUE tasks')
    parser.add_argument('--dset', type=str, help="Dataset name")
    parser.add_argument('--model_name', type=str, default='roberta-base', help="Pre-trained RoBERTa model")
    parser.add_argument('--save_total_limit', type=int, default=1, help="Limit the total number of checkpoints")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for initialization")

    parser.add_argument('--training_dataset', default='', type=str)
    parser.add_argument('--train_size', type=int, default=0)

    parser.add_argument('--validation_dataset', default='', type=str)
    # parser.add_argument('--val_size', type=int, default=-1)

    parser.add_argument('--max_epoch', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training") #16 can go up to 128
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate") # 1e-5, 1e-2
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for the optimizer")

    args = parser.parse_args()

    if args.dset in ['cola', 'mnli', 'qnli', 'rte', 'qqp', 'sst2', 'sts-b']:
        if args.training_dataset == '':
            args.training_dataset = args.dset + '_SAE_train'
        
        if args.validation_dataset == '':
            args.validation_dataset = args.dset + '_SAE_validation'
                    
    args.filename = args.training_dataset + '_' + str(args.train_size)

    ROOT = f"./output/{args.filename}/train"
    args.logging_dir = ROOT + "/log"
    args.result_dir = ROOT + "result"
    args.netB_dir = ROOT + "netB"
    args.netC_dir = ROOT + "netC"

    for path in [args.logging_dir, args.result_dir, args.netB_dir, args.netC_dir]:
        if os.path.isfile(path):
            raise Exception(f"File already exists at {path}") 

    set_seed(args.seed)
    args.num_input, args.class_num = get_class(args.dset)

    # Looping over the target domains to perform adaptation for each:        
    trainer(args, accelerator)