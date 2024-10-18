# UPDATE LOGGING AND SAVING MODELS
# SAVE ALL THE PARAMETERS FOR EACH TRAINING ITERATION
# training logs are not reported

import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
# from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import time

import evaluate

from transformers import RobertaTokenizer, Trainer, TrainingArguments, RobertaForSequenceClassification
from datasets import load_from_disk

from helper import get_class # RobertaForCustomClassification

# output_hidden_states=True

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def obtain_label(inputs_, model, args):
    with torch.no_grad():
        loader = inputs_

        inputs = inputs_['input_ids'].to(args.device)

        # ACTUAL LABEL
        labels = inputs_['labels'].to(args.device)

        raw_outputs = model(inputs)

        all_fea = raw_outputs.hidden_states[-1][:, 0, :].float()#.cpu()

        # PREDICTED LABEL
        all_output = raw_outputs.logits.float()#.cpu()
        all_label = labels.float()

    all_output = nn.Softmax(dim=1)(all_output)

    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
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
    for _ in range(2):
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
    print("ARE TWO PREIDCT EQUAL: ", (old_predict==predict).all())

    # THIS SHOULD BE DIFFERENT BUT POSSIBLE THAT IT IS NOT DIFFERENT INTERVAL NEEDS TO INCREASE
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(acc_pre_tta * 100, acc_post_tta * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return predict.astype('int')

class CustomTrainer(Trainer):
    def __init__(self, extra_args = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # can incorporate this into args or kwargs
        # self.device = self.args.device
        # self.interval = self.args.interval
        # print(self.args)
        # exit(0)
        self.extra_args = extra_args
        self.device = extra_args.device
        self.interval_iter = self.args.num_train_epochs // extra_args.interval

        # self.mem_label = None

    def compute_loss(self, model, inputs, return_outputs=False):
        # current_iter = self.state.global_step
        # print(current_iter)
        # if current_iter % self.interval_iter == 0 and self.extra_args.cls_par > 0:
            # pseudo_label
        model.eval()
        mem_label = obtain_label(inputs, model, self.extra_args)
        mem_label = torch.from_numpy(mem_label).to(self.device)
        model.train()

        inputs = {k: v.to(self.device) for k, v in inputs.items()} #  if k in ["input_ids", "attention_mask", "label"]

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Custom loss calculation
        classifier_loss = nn.CrossEntropyLoss()(logits, mem_label)

        classifier_loss *= self.extra_args.cls_par
        if self.extra_args.ent:
            softmax_out = nn.Softmax(dim=1)(logits)
            entropy_loss = torch.mean(Entropy(softmax_out))
            if self.extra_args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + self.extra_args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * self.extra_args.ent_par
            classifier_loss += im_loss

        # return (classifier_loss, logits) if return_outputs else classifier_loss

        # https://discuss.huggingface.co/t/evalprediction-has-an-unequal-number-of-label-ids-and-predictions/10557/4
        return (classifier_loss, {"label": logits}) if return_outputs else classifier_loss

def train_target(args):
    # Load GLUE dataset
    dataset = load_from_disk(f"./datasets/{args.dset}_validation")#.select(range(20))

    args.savename = 'par_' + str(args.cls_par)
    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')

    # Load Roberta Model and Tokenizer
    # args.model_name
    tokenizer = RobertaTokenizer.from_pretrained('./models/train/roberta_tokenizer_rte')

    model = RobertaForSequenceClassification.from_pretrained('./models/train/roberta_model_rte', output_hidden_states=True)
    args.device = torch.device(f'cuda:0')
    model.to(args.device)

    # model = nn.DataParallel(RobertaForSequenceClassification.from_pretrained('./models/train/roberta_model_rte', output_hidden_states=True), device_ids=[1,2])
    # args.device = model.device

    # Freeze the model here
    param_group = []
    for name, param in model.classifier.named_parameters():
        param.requires_grad = False
        if args.lr_decay1 > 0:
            param_group += [{'params': param, 'lr': args.lr * args.lr_decay1}]

    def tokenize_function_one_output(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True)

    def tokenize_function_two_outputs(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

    if args.num_input == 1:
        tokenize_function = tokenize_function_one_output
    elif args.num_input == 2:
        tokenize_function = tokenize_function_two_outputs
    else:
        raise Exception(f'{args.num_input} is an invalid number of inputs')

    # Set format for PyTorch
    encoded_dataset = dataset.map(tokenize_function, batched=True)
    encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # print(encoded_dataset[0])

    # Training Arguments
    training_args = TrainingArguments(
        output_dir='./results/test',
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

    # Define Trainer
    # THIS IS NOT FOR QNLI or WHATEVER THE REGRESSION TASK IS
    metric = evaluate.load("glue", args.dset)
    # metric = load_metric("glue", args.dset)
    def compute_metrics(eval_pred):
        outputs, labels = eval_pred
        #outputs: first item is logits, then hidden states, then attention_weights (if this is enabled)
        # predictions = torch.argmax(torch.tensor(outputs[0]), dim=-1)
        predictions = torch.argmax(torch.tensor(outputs), dim=-1)

        return metric.compute(predictions=predictions, references=labels)

    trainer = CustomTrainer(
        # device = args.device,
        # interval = args.interval,

        extra_args=args,
        model=model,
        args=training_args,
        train_dataset=encoded_dataset,
        eval_dataset=encoded_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate the model
    trainer.train()
    trainer.evaluate()

    model.save_pretrained(f"./models/test/roberta_model_{args.dset}_fullbatch")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT adaptation for RoBERTa on GLUE tasks')
    parser.add_argument('--gpu_id', type=str, default='2', help="GPU id to use")
    parser.add_argument('--dset', type=str, default='rte', choices=["sst2", "mnli", "mrpc", "cola", "stsb", "qqp", "rte", "wnli", "qnli"], help="GLUE task name")

    # This needs to be the trained model
    parser.add_argument('--model_name', type=str, default='roberta-base', help="Pre-trained RoBERTa model")
    parser.add_argument('--model_output_size', type=int, default=768, help='The size of the output of your model, RoBERTA would be 768.')
    parser.add_argument('--max_epoch', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--interval', type=int, default=15)

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--threshold', type=int, default=0)
   
    parser.add_argument('--max_length', type=int, default=128, help="Maximum sequence length")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for the optimizer")
    parser.add_argument('--logging_dir', type=str, default=f"./logs/test/{int(time.time())}", help="Directory for storing logs")
    parser.add_argument('--logging_steps', type=int, default=500, help="Steps for logging")
    parser.add_argument('--save_total_limit', type=int, default=1, help="Limit the total number of checkpoints")
    parser.add_argument('--output_dir', type=str, default='./results', help="Directory for saving model checkpoints")
    
    parser.add_argument('--seed', type=int, default=42, help="Random seed for initialization")
    
    args = parser.parse_args()

    # Set environment variables and seeds
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    set_seed(args.seed)
    # Optionally, `torch.backends.cudnn.deterministic = True` can be uncommented to further control reproducibility.
        
    # 'cola', 'mnli', 'qnli', 'rte', 'qqp', 'sst2', 'sts-b', 'all'

    # # if torch.backends.mps.is_built:
    # #     args.device = torch.device('mps')
    # if torch.cuda.is_available():
    #     # args.device = torch.cuda.current_device()
    #     # args.device = torch.device("cuda:3,4")
    #     args.device = torch.device('cuda:3')
    #     # args.device = 'parallel'
    # else:
    #     args.device = torch.device("cpu")

    args.num_input, args.class_num = get_class(args.dset)
    
    # Looping over the target domains to perform adaptation for each:        
    train_target(args)