import dataclasses
import random
import numpy as np

import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from transformers import TrainingArguments

@dataclasses.dataclass
class TestCustomTrainingArguments(TrainingArguments):
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

    acc_post_tta = np.sum(predict == all_label.float().cpu().numpy()) / len(all_fea)

    print("Embedding is updated: ", (old_predict==predict).all())

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

# NetB
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.5):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
        out, _ = self.gru(x, h0)
        return out

# NetC
class Classifier(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_rate=0.5):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # First layer
        self.fc2 = nn.Linear(hidden_size, output_size)      # Second layer

    def forward(self, x):
        x = self.fc1(x[:, -1, :])
        x = nn.ReLU()(x)               # Activation function
        out = self.fc2(x)
        return out