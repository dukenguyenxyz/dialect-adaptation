import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

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