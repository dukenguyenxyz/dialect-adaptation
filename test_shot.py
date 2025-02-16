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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import RobertaModel, RobertaTokenizer, Trainer
import evaluate
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator

from helper import get_class, set_seed, get_metric, MultiModelWrapper, BiGRU, Classifier, Entropy, obtain_label, TestCustomTrainingArguments

DATASET_PATH = ''

class CustomTrainer(Trainer):
    def __init__(self, out_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_file = out_file

        self.stats = {
            'improved': 0,
            'degraded': 0,
            'no_change':0
        }

    def compute_loss(self, model, inputs, return_outputs=False):
        model.eval()
        mem_label, self.stats = obtain_label(inputs, model, self.args, self.out_file, self.stats)
        mem_label = torch.from_numpy(mem_label).to(self.args.device)
        model.train()

        inputs = {k: v.to(self.args.device) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}

        # Forward pass
        outputs = model(inputs)

        # Custom loss calculation
        classifier_loss = nn.CrossEntropyLoss()(outputs, mem_label)

        classifier_loss *= self.args.cls_par
        if self.args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs)
            entropy_loss = torch.mean(Entropy(softmax_out, self.args.epsilon))
            if self.args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + self.args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * self.args.ent_par
            classifier_loss += im_loss

        # return (classifier_loss, outputs) if return_outputs else classifier_loss

        # https://discuss.huggingface.co/t/evalprediction-has-an-unequal-number-of-label-ids-and-predictions/10557/4
        return (classifier_loss, {"label": outputs}) if return_outputs else classifier_loss

def tester(args, accelerator):
    device = torch.device('cuda')

    # Training Arguments
    training_args = TestCustomTrainingArguments(
        output_dir=args.result_dir,
        eval_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        label_names=["labels"],
        num_train_epochs=args.max_epoch,
        weight_decay=args.weight_decay,
        # logging_dir=args.logging_dir,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,

        class_num = args.class_num,

        ent = args.ent,
        gent = args.gent,
        ent_par = args.ent_par,
        cls_par = args.cls_par,
        epsilon = args.epsilon,
        threshold = args.threshold,
        distance = args.distance,
    )

    if accelerator.is_main_process:
        # Initialize TensorBoard SummaryWriter
        writer = SummaryWriter(log_dir=args.logging_dir)    
        
        # Log all hyperparameters using SummaryWriter for TensorBoard
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

    netB = BiGRU(input_size=netF.config.hidden_size, hidden_size=mid_hidden_size)
    netB.load_state_dict(torch.load(args.in_netB_dir + '.pth'))
    netB.to(device)
    
    netC = Classifier(hidden_size=mid_hidden_size, output_size=args.class_num)
    netC.load_state_dict(torch.load(args.in_netC_dir + '.pth'))
    netC.to(device)
    
    for k, v in netF.named_parameters():
        v.requires_grad = False
    
    model = MultiModelWrapper(netF, netB, netC)

    if args.dset == 'besstie':
        dataset = load_dataset("json", data_files=f"{DATASET_PATH}/BESSTIE-sentiment/valid/{args.validation_dataset}.jsonl")['train']
        dataset = dataset.rename_column("text", "sentence")
        dataset = dataset.rename_column("sentiment_label", "labels")
        dataset = dataset.remove_columns('id')
    
    elif args.dset == 'amazon_text':
        dataset = load_dataset('json', data_files=f'{DATASET_PATH}/amazon_text/{args.validation_dataset}.json', split='train')            
        dataset = dataset.rename_column("review", "sentence")
        dataset = dataset.remove_columns(["summary", "helpful"])
        dataset = dataset.rename_column("label", "labels")

    elif args.dset in ['cola', 'mnli', 'qnli', 'rte', 'qqp', 'sst2', 'sts-b']:
        if args.validation_dataset:
            dataset = load_from_disk(f'{DATASET_PATH}/multivalue/{args.validation_dataset}')    
        else:
            dataset = load_dataset("glue", args.dset, split='validation')
            args.validation_dataset = args.dset + '_SAE_validation'

        dataset = dataset.rename_column("label", "labels")

    if args.val_size > 0:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.val_size))

    def tokenize_function(examples):
        if args.dset == "rte":
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)
        elif args.dset == "qnli":
            return tokenizer(examples["question"], examples["sentence"], padding="max_length", truncation=True)
        else:
            return tokenizer(examples["sentence"], padding="max_length", truncation=True)

    ## TOKENIZER
    encoded_dataset = dataset.map(tokenize_function, batched=True)
    encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    if args.dset in ['cola', 'mnli', 'qnli', 'rte', 'qqp', 'sst2', 'sts-b']:
        metric = evaluate.load('glue', args.dset)
    else:
        metric = evaluate.load('accuracy')
    compute_metrics = get_metric(metric)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    num_training_steps = len(encoded_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs

    def lr_lambda(current_step):
        p = current_step / num_training_steps
        return (1 + 10 * p) ** -0.75

    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step))

    trainer = CustomTrainer(
        out_file=args.out_file,
        model=model,
        args=training_args,
        train_dataset=encoded_dataset,
        eval_dataset=encoded_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler)
    )

    # Train and evaluate the model
    trainer.train()

    torch.save(model.netB.state_dict(), args.out_netB_dir + '.pth')
    torch.save(model.netC.state_dict(), args.out_netC_dir + '.pth')

    # Evaluate the model on the full validation dataset after training
    final_eval_results = trainer.evaluate(eval_dataset=encoded_dataset)

    if args.dset == 'cola':
        metric_name = 'eval_matthews_correlation'
    elif args.dset == 'sts-b':
        metric_name = 'eval_pearson'
    else:
        metric_name = 'eval_accuracy'

    # Log the final evaluation metrics explicitly with a custom step or labels
    trainer.log({
        "eval/final_accuracy": final_eval_results[metric_name],  # Example, log additional metrics as needed
        "eval/final_loss": final_eval_results['eval_loss']
    })

    # Optionally, you can print or save the final evaluation metrics
    print("Final Evaluation Results:", final_eval_results)

    if accelerator.is_main_process:
        # Add result to csv file
        # Define the path to the CSV file
        csv_file = "test_shot.csv"

        # Check if the CSV file exists
        if os.path.exists(csv_file):
            # Load the existing CSV into a DataFrame
            df = pd.read_csv(csv_file)
            print("CSV loaded successfully.")
        else:
            # Define column names and data types
            column_types = {
                "In_Filename": str,
                "Out_Filename": str,
                "Dataset": str,
                "Epoch": int,            # Integer type for epoch numbers
                "Eval": str,         # String type for evaluation data
                "Seed": int,
                "Acc": float,
                "weight_decay": float,
                "ent_par": float,
                "cls_par": float,
                "epsilon": float,
                "threshold": float,
                "Val_size": int
            }

            # Initialize DataFrame with column names and types
            df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in column_types.items()})
            print("CSV not found. Initialized a new DataFrame.")

        new_data = {
            "In_Filename": args.int_filename, 
            "Out_Filename": args.out_filename, 
            "Dataset": args.dset, 
            "Epoch": args.max_epoch, 
            "Eval": args.validation_dataset, 
            "Seed": args.seed, 
            "Acc": final_eval_results['eval_accuracy'],
            "weight_decay": args.weight_decay,
            "ent_par": args.ent_par,
            "cls_par": args.cls_par,
            "epsilon": args.epsilon,
            "threshold": args.threshold,
            "Val_size": args.val_size
        }

        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

        # Save the DataFrame to the CSV file
        df.to_csv(csv_file, index=False)
        print(f"Data saved to {csv_file}.")

if __name__ == "__main__":
    accelerator = Accelerator()
    
    parser = argparse.ArgumentParser(description='SHOT adaptation for RoBERTa on GLUE tasks')
    parser.add_argument('--gpu_id', type=str, default='0', help="GPU id to use")
    parser.add_argument('--dset', type=str, default='sst2', help="GLUE task name")
    parser.add_argument('--model_name', type=str, default='roberta-base', help="Pre-trained RoBERTa model")

    parser.add_argument('--save_total_limit', type=int, default=1, help="Limit the total number of checkpoints")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for initialization")
    parser.add_argument('--validation_dataset', default='', type=str)
    parser.add_argument('--val_size', type=int, default=0)

    parser.add_argument('--int_filename', type=str)

    parser.add_argument('--max_epoch', type=int, default=50, help="Number of training epochs")
    # parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")

    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate") # 1e-5
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for the optimizer")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3) # 0.3
    parser.add_argument('--ent_par', type=float, default=1.0) # 1.0
    
    # 0.3 really bad, 0.8 a lot better
    parser.add_argument('--epsilon', type=float, default=1e-5) # 1e-5
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    
    args = parser.parse_args()

    IN_ROOT = f"./output/{args.int_filename}/train"    
    # IN DIR
    args.in_netB_dir = IN_ROOT + "netB"
    args.in_netC_dir = IN_ROOT + "netC"
    
    # out_filename = int(time.time())
    args.out_filename = args.validation_dataset + str(args.val_size)
    
    OUT_ROOT = f"./output/{args.int_filename}/test/{args.out_filename}/"
    args.logging_dir = OUT_ROOT + "log"
    args.result_dir = OUT_ROOT + "result"
    
    # OUT DIR
    args.out_netB_dir = OUT_ROOT + "netB"
    args.out_netC_dir = OUT_ROOT + "netC"
    
    args.savename = 'par_' + str(args.cls_par)
    
    if not os.path.exists(OUT_ROOT + "log"):
        os.makedirs(OUT_ROOT + "log")
    
    args.out_file = open(OUT_ROOT + f'log/log_{args.savename}.txt', 'w')

    for path in [args.logging_dir, args.result_dir, args.out_netB_dir, args.out_netC_dir]:
        if os.path.isfile(path):
            raise Exception(f"File already exists at {path}") 

    set_seed(args.seed)
    args.num_input, args.class_num = get_class(args.dset) # 1, 2

    tester(args, accelerator)