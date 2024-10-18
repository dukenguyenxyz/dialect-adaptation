import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_from_disk
import evaluate

task = 'rte'

train_dataset = load_dataset("glue", task, split='train').select(range(20))

#.select(range(20))

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2) #output_hidden_states=True

def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)

validation_dataset = load_from_disk(f"./datasets/{task}_validation").select(range(20))
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

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=50,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

model.save_pretrained(f"./models/train/roberta_model_{task}")
tokenizer.save_pretrained(f"./models/train/roberta_tokenizer_{task}")

# Evaluate the model
trainer.evaluate()