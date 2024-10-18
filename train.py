import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, load_from_disk
import evaluate
import os
import time

task = 'rte'

train_dataset = load_dataset("glue", task, split='train')#.select(range(20))

#.select(range(20))

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2) #output_hidden_states=True

# device_ids = [0,1]
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# model = torch.nn.DataParallel(model, device_ids=device_ids)
# model.to(f'cuda:{device_ids[0]}')
# model.cuda(device_ids[0])

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device(f'cuda:0')
model.to(device)

def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)

validation_dataset = load_from_disk(f"./datasets/{task}_validation")#.select(range(20))
tokenized_validation_datasets = validation_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_train_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Split the dataset into training and validation sets
train_dataset = tokenized_train_datasets
validation_dataset = tokenized_validation_datasets

# Load the metric for evaluation
# metric = load_metric("glue", task)
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return metric.compute(predictions=predictions, references=labels)

class CustomTrainer(Trainer):
    def __init__(self, label_smoothing=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = {k: v.to(model.device) for k, v in inputs.items()} #  if k in ["input_ids", "attention_mask", "label"]

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        # logits = outputs

        # Custom loss calculation
        classifier_loss = self.loss(logits, inputs['labels'])

        # https://discuss.huggingface.co/t/evalprediction-has-an-unequal-number-of-label-ids-and-predictions/10557/4
        return (classifier_loss, {"label": logits}) if return_outputs else classifier_loss

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir=f"./logs/train/{int(time.time())}",
    logging_steps=10,
)

# Initialize the Trainer
trainer = CustomTrainer(
    label_smoothing=0.1,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

# Train the model
trainer.train()

model.save_pretrained(f"./models/train/roberta_model_{task}")
tokenizer.save_pretrained(f"./models/train/roberta_tokenizer_{task}")

# Evaluate the model
trainer.evaluate()