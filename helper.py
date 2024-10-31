import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
import numpy as np
import random

def get_class(dset):
    if dset == 'cola':
        num_input = 1
        class_num = 2
    elif dset == 'mnli':
        num_input = 2
        class_num = 3
    elif dset == 'qnli':
        num_input = 2
        class_num = 2
    
    # THIS IS THE SMALLEST ONE
    elif dset == 'rte':
        num_input = 2
        class_num = 2
    elif dset == 'qqp':
        num_input = 2
        class_num = 2
    elif dset == 'sst2':
        num_input = 1
        class_num = 2
    elif dset == 'sts-b':
        num_input = 2
        regression_range = [0, 5]
        return (num_input, regression_range)
    
    return (num_input, class_num)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_tokenize_function(num_input, tokenizer):
    def tokenize_function_one_output(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True)

    def tokenize_function_two_outputs(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

    if num_input == 1:
        tokenize_function = tokenize_function_one_output
    elif num_input == 2:
        tokenize_function = tokenize_function_two_outputs
    else:
        raise Exception(f'{num_input} is an invalid number of inputs')
    
    return tokenize_function

def get_metric(metric):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        return metric.compute(predictions=predictions, references=labels)

    return compute_metrics