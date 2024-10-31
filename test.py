import argparse
import os
import time
import dataclasses

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist

from transformers import RobertaTokenizer, Trainer, TrainingArguments, RobertaForSequenceClassification
from torch.utils.tensorboard import SummaryWriter
import evaluate
from datasets import load_from_disk

from helper import get_class, set_seed, get_tokenize_function, get_metric

def Entropy(input_, epsilon):
    # bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def obtain_label(inputs_, model, args, out_file, stats):
    with torch.no_grad():
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

@dataclasses.dataclass
class CustomTrainingArguments(TrainingArguments):
    model_name: str = ''
    dset: str = 'rte'
    class_num: int = 2
    num_input: int = 2
    out_filename: str = ''

    ent: bool = True
    gent: bool = True
    ent_par: float = 1.0
    cls_par: float = 0.3
    epsilon: float = 1e-5
    lr_decay1 : float = 0.1
    threshold : int = 0
    distance : str = 'cosine'

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

        inputs = {k: v.to(self.args.device) for k, v in inputs.items()} #  if k in ["input_ids", "attention_mask", "label"]

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Custom loss calculation
        classifier_loss = nn.CrossEntropyLoss()(logits, mem_label)

        classifier_loss *= self.args.cls_par
        if self.args.ent:
            softmax_out = nn.Softmax(dim=1)(logits)
            entropy_loss = torch.mean(Entropy(softmax_out, self.args.epsilon))
            if self.args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + self.args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * self.args.ent_par
            classifier_loss += im_loss

        # return (classifier_loss, logits) if return_outputs else classifier_loss

        # https://discuss.huggingface.co/t/evalprediction-has-an-unequal-number-of-label-ids-and-predictions/10557/4
        return (classifier_loss, {"label": logits}) if return_outputs else classifier_loss

def tester(args):
    # Training Arguments
    training_args = CustomTrainingArguments(
        model_name = args.model_name,
        dset = args.dset,
        class_num = args.class_num,
        num_input = args.num_input,
        out_filename = args.out_filename,

        ent = args.ent,
        gent = args.gent,
        ent_par = args.ent_par,
        cls_par = args.cls_par,
        epsilon = args.epsilon,
        lr_decay1 = args.lr_decay1,
        threshold = args.threshold,
        distance = args.distance,

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

    tokenizer = RobertaTokenizer.from_pretrained(args.in_token_dir)
    model = RobertaForSequenceClassification.from_pretrained(args.in_model_dir, output_hidden_states=True)

    # Freeze the model here
    param_group = []
    for name, param in model.classifier.named_parameters():
        param.requires_grad = False
        if args.lr_decay1 > 0:
            param_group += [{'params': param, 'lr': args.lr * args.lr_decay1}]

    ## MODEL
    args.device = torch.device(f'cuda:0')
    model.to(args.device)

    # Load GLUE dataset
    test_size = 32

    dataset = load_from_disk(args.validation_dataset)#.select(range(test_size))
    ## TOKENIZER
    tokenize_function = get_tokenize_function(args.num_input, tokenizer)
    encoded_dataset = dataset.map(tokenize_function, batched=True)
    encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    metric = evaluate.load("glue", args.dset)
    compute_metrics = get_metric(metric)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=training_args.logging_dir)

    # Log all hyperparameters using SummaryWriter for TensorBoard
    # training_args.to_dict() converts the hyperparameters to a dictionary
    for key, value in training_args.to_dict().items():
        writer.add_text(f"hyperparameters/{key}", str(value), 0)

    # Close the SummaryWriter after logging all the parameters
    writer.close()

    trainer = CustomTrainer(
        out_file=args.out_file,
        model=model,
        args=training_args,
        train_dataset=encoded_dataset,
        eval_dataset=encoded_dataset,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate the model
    trainer.train()

    model.save_pretrained(args.out_model_dir)


    # Evaluate the model
    # trainer.evaluate()
    # Evaluate the model on the full validation dataset after training
    final_eval_results = trainer.evaluate(eval_dataset=encoded_dataset)

    # Log the final evaluation metrics explicitly with a custom step or label
    trainer.log({
        "eval/final_accuracy": final_eval_results['eval_accuracy'],  # Example, log additional metrics as needed
        "eval/final_loss": final_eval_results['eval_loss']
    })

    # Optionally, you can print or save the final evaluation metrics
    print("Final Evaluation Results:", final_eval_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT adaptation for RoBERTa on GLUE tasks')
    parser.add_argument('--gpu_id', type=str, default='2', help="GPU id to use")
    parser.add_argument('--dset', type=str, default='rte', choices=["sst2", "mnli", "mrpc", "cola", "stsb", "qqp", "rte", "wnli", "qnli"], help="GLUE task name")
    parser.add_argument('--model_name', type=str, default='roberta-base', help="Pre-trained RoBERTa model")
    parser.add_argument('--logging_steps', type=int, default=10, help="Steps for logging")
    parser.add_argument('--save_total_limit', type=int, default=1, help="Limit the total number of checkpoints")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for initialization")
    parser.add_argument('--validation_dataset', type=str)

    parser.add_argument('--max_epoch', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate") # 1e-5
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for the optimizer")

    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent_par', type=float, default=1.0) # 1.0

    # 0.3 really bad, 0.8 a lot better
    parser.add_argument('--cls_par', type=float, default=0.3) # 0.3
    parser.add_argument('--epsilon', type=float, default=1e-5) # 1e-5
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--threshold', type=int, default=0)

    parser.add_argument('--out_filename', type=str, default='')

    # parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    
    args = parser.parse_args()

    if not args.out_filename:
        args.out_filename = args.model_name + '_' + args.dset + '_' + str(int(time.time()))

    args.int_filename = 'roberta-base_rte_1730199414'

    args.in_model_dir = f"./output/models/train/{args.int_filename}_model"
    args.in_token_dir = f"./output/models/train/{args.int_filename}_tokenizer"

    args.out_model_dir = f"./output/models/test/{args.out_filename}_model"

    args.logging_dir = f"./output/logs/test/{args.out_filename}"
    args.result_dir = f"./output/results/test/{args.out_filename}"
    # args.validation_dataset = f"./datasets/{args.dset}_validation_indian"

    if not os.path.exists(f'./logs/test/{args.out_filename}'):
        os.makedirs(f'./logs/test/{args.out_filename}')
    args.savename = 'par_' + str(args.cls_par)
    args.out_file = open(f'./logs/test/{args.out_filename}/log_{args.savename}.txt', 'w')

    # Optionally, `torch.backends.cudnn.deterministic = True` can be uncommented to further control reproducibility.
    set_seed(args.seed)
    args.num_input, args.class_num = get_class(args.dset)

    # if torch.backends.mps.is_built:
    #     args.device = torch.device('mps')
    # if torch.cuda.is_available():
    #     args.device = torch.device('cuda:3')
    # else:
    #     args.device = torch.device("cpu")

    # Set environment variables and seeds
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # Looping over the target domains to perform adaptation for each:        
    tester(args)