# Train both the mlm and the nsp objective simulataneously
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel
from torch.optim import AdamW

from nsp import prepare_nsp_dataset
from mlm import prepare_mlm_dataset

from datasets import load_dataset

def train_ss_model(datasets, model_name='roberta-base', batch_size=8, num_epochs=30, learning_rate=5e-5):
    nsp_dataloader, mlm_dataloader = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in datasets]

    # Load pre-trained RoBERTa model
    backbone_model = RobertaModel.from_pretrained(model_name)
    nsp_classifier_layer = nn.Linear(backbone_model.config.hidden_size, 2)
    mlm_classifier_layer = nn.Linear(backbone_model.config.hidden_size, backbone_model.config.vocab_size)
    
    optimizer = AdamW(list(backbone_model.parameters()) + list(nsp_classifier_layer.parameters()) + list(mlm_classifier_layer.parameters()), lr=learning_rate)
    loss_fct = nn.CrossEntropyLoss()

    # Training loop
    backbone_model.train()
    nsp_classifier_layer.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_nsp, batch_mlm in zip(nsp_dataloader, mlm_dataloader):
            # Process one item from each dataset
            optimizer.zero_grad()
            
            ## NSP Process
            outputs = backbone_model(input_ids=batch_nsp['input_ids'].squeeze(), attention_mask=batch_nsp['attention_mask'].squeeze())
            sequence_output = outputs[0]
            pooled_output = sequence_output[:, 0, :]  # Take the representation of the [CLS] token
            logits = nsp_classifier_layer(pooled_output)
            loss_nsp = loss_fct(logits.view(-1, 2), batch_nsp['labels'].squeeze().view(-1))

            ## MLM Process
            outputs = backbone_model(input_ids=batch_mlm['input_ids']) #, attention_mask=batch_mlm['attention_mask']
            sequence_output = outputs[0]
            prediction_scores = mlm_classifier_layer(sequence_output)
            prediction_scores = prediction_scores.view(-1, backbone_model.config.vocab_size)
            labels = batch_mlm['labels'].view(-1)
            loss_mlm = loss_fct(prediction_scores, labels)

            loss = loss_nsp + loss_mlm
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / min(len(nsp_dataloader), len(mlm_dataloader))
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

if __name__ == "__main__":

    # documents = [
    #     "The weather is nice today. I plan to go for a walk. There are many flowers in bloom.",
    #     "Artificial intelligence is a growing field. It has applications in many areas. The future is promising."
    # ]

    dataset = load_dataset("json", data_files=f"datasets/BESSTIE-sentiment/valid/reddit-sentiment-in.jsonl")['train']
    dataset = dataset.rename_column("text", "sentence")
    dataset = dataset.rename_column("sentiment_label", "label")
    dataset = dataset.remove_columns('id')

    model_name='roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    # Can combine into one dataset with 6 fields and then into dataloader
    nsp_dataset = prepare_nsp_dataset(dataset['sentence'], tokenizer)
    mlm_dataset = prepare_mlm_dataset(dataset['sentence'], tokenizer)

    # print(len(nsp_dataset))
    # print(len(mlm_dataset))

    train_ss_model([nsp_dataset, mlm_dataset], model_name)