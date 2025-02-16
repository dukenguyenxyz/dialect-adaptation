import torch
import torch.nn as nn
import numpy as np
import random
from scipy.spatial.distance import cdist
import dataclasses
from transformers import TrainingArguments

from transformers import RobertaModel, RobertaConfig

@dataclasses.dataclass
class TestCustomTrainingArguments(TrainingArguments):
    # output_dir: str = ''
    # eval_strategy: str = 'epoch'
    # learning_rate: float = 0
    # per_device_train_batch_size: int = 16
    # per_device_eval_batch_size: int = 16
    # num_train_epochs: int = 30
    # weight_decay: float = 0.01
    # logging_dir: str = './'
    # logging_steps: int = 10
    # save_total_limit: int = 1

    class_num: int = 2

    ent: bool = True
    gent: bool = True
    ent_par: float = 1.0
    cls_par: float = 0.3
    epsilon: float = 1e-5

    threshold: int = 0
    distance: str = 'cosine'
    def __post_init__(self):
            super().__post_init__()  # Ensure parent initialization is handled

def Entropy(input_, epsilon):
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def obtain_label(inputs_, model, args, out_file, stats):
    with torch.no_grad():
        # inputs = inputs_['input_ids'].to(args.device)

        # ACTUAL LABEL
        labels = inputs_['labels'].to(args.device)

        all_fea, all_output = model(inputs_, feat=True)
        all_label = labels.float()

    all_output = nn.Softmax(dim=1)(all_output)

    # ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    # unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    old_predict = predict.detach().clone().cpu().numpy()

    # LABEL
    acc_pre_tta = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1, device = all_fea.device)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    K = all_output.size(1)

    all_fea = all_fea.float().cpu().numpy()
    # all_fea = all_fea.float().numpy()
    aff = all_output.float().cpu().numpy()
    # aff = all_output.float().numpy()

    # re-labelling via centroids
    for _ in range(5):
        # initc is the centroid for each class
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        if torch.is_tensor(predict):
            predict = predict.cpu()
        cls_count = np.eye(K)[predict]
        cls_count = cls_count.sum(axis=0)
        labelset = np.where(cls_count>args.threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    # ASSIGN NEW LABEL
    acc_post_tta = np.sum(predict == all_label.float().cpu().numpy()) / len(all_fea)

    # THEY ARE FREAKING EQUAL
    print("Embedding is updated: ", (old_predict==predict).all())

    # THIS SHOULD BE DIFFERENT BUT POSSIBLE THAT IT IS NOT DIFFERENT INTERVAL NEEDS TO INCREASE
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(acc_pre_tta * 100, acc_post_tta * 100)

    if acc_post_tta > acc_pre_tta:
        print('Improved ->')
        stats['improved'] += 1
    elif acc_post_tta < acc_pre_tta:
        print('Degraded <-')
        stats['degraded'] += 1
    else:
        print('No change')
        stats['no_change'] += 1

    out_file.write(log_str + '\n')
    out_file.flush()
    print(log_str+'\n')

    print('Improved: {}. Degraded: {}. No change: {}'.format(stats['improved'], stats['degraded'], stats['no_change']))

    return predict.astype('int'), stats

class MultiModelWrapper(nn.Module):
    def __init__(self, netF, netB, netC):
        super().__init__()
        self.netF = netF
        self.netB = netB
        self.netC = netC

    def forward(self, inputs, feat=False):
        feature = self.netB(self.netF(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).last_hidden_state)
        out = self.netC(feature)
        if feat:
            return feature[:, -1, :], out
        return out

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
    
    elif dset == 'besstie':
        num_input = 1
        class_num = 2
    
    elif dset == "amazon_text":
        num_input = 1
        class_num = 2
    
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
        out = metric.compute(predictions=predictions, references=labels)
        return out

    return compute_metrics

# Calculate distance:
# Get 50 random entry from one dialect vs another 50 random entry from another dialect
# Get the average pairwise cosine distance????

# NetB
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.5):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        # self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
        out, _ = self.gru(x, h0)
        # out = self.batch_norm(out)  # Apply Batch Normalization
        return out

# NetC
# class Classifier(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super(Classifier, self).__init__()
#         self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional
        
#     def forward(self, x):
#         out = self.fc(x[:, -1, :])
#         return out

class Classifier(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_rate=0.5):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # First layer
        # self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)      # Second layer

    def forward(self, x):
        x = self.fc1(x[:, -1, :])
        x = nn.ReLU()(x)               # Activation function
        # x = self.dropout(x)            # Apply Dropout after activation
        out = self.fc2(x)
        return out

# Example usage:

# model_name = 'roberta-base'
# netF = RobertaModel.from_pretrained(model_name)
# mid_hidden_size = netF.config.hidden_size // 2 # 256

# netB = BiGRU(input_size=netF.config.hidden_size, hidden_size=mid_hidden_size)
# netC = Classifier(hidden_size=mid_hidden_size, output_size=2)

# for k, v in netC.named_parameters():
#     v.requires_grad = False

# param_group = []
# for k, v in netF.named_parameters():
#     if args.lr_decay1 > 0:
#         param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
#     else:
#         v.requires_grad = False
# for k, v in netB.named_parameters():
#     if args.lr_decay2 > 0:
#         param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
#     else:
#         v.requires_grad = False
