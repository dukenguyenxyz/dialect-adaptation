
import random
import re

import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel, AdamW

def prepare_nsp_dataset(documents, tokenizer):
    def create_nsp_dataset(documents):
        pairs = []
        for doc in documents:
            # Split document into sentences using full stops, exclamation marks, and question marks
            sentences = re.split(r'[.!?] ', doc.strip())
            for i in range(len(sentences) - 1):
                # Positive example (next sentence relationship)
                pairs.append((sentences[i], sentences[i + 1], 1))
                
                # Negative example (random sentence from another document)
                if len(documents) > 1:
                    random_doc = random.choice([d for d in documents if d != doc])
                    random_sentence = random.choice(re.split(r'[.!?] ', random_doc.strip()))
                    pairs.append((sentences[i], random_sentence, 0))
        return pairs

    def tokenize_pair(pair, tokenizer):
        sent1, sent2, label = pair
        encoded = tokenizer(
            sent1, 
            sent2, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=128  # Adjust this based on your needs
        )
        encoded['labels'] = label
        return encoded

    pairs = create_nsp_dataset(documents)
    tokenized_data = [tokenize_pair(pair, tokenizer) for pair in pairs]
    return tokenized_data

def train_nsp_model(dataset, model_name='roberta-base', batch_size=8, num_epochs=30, learning_rate=5e-5):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load pre-trained RoBERTa model
    backbone_model = RobertaModel.from_pretrained(model_name)
    nsp_classifier_layer = nn.Linear(backbone_model.config.hidden_size, 2)
    optimizer = AdamW(list(backbone_model.parameters()) + list(nsp_classifier_layer.parameters()), lr=learning_rate)
    loss_fct = nn.CrossEntropyLoss()

    # Training loop
    backbone_model.train()
    nsp_classifier_layer.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].squeeze()
            attention_mask = batch['attention_mask'].squeeze()
            labels = batch['labels'].squeeze()

            outputs = backbone_model(input_ids, attention_mask=attention_mask)
            sequence_output = outputs[0]
            pooled_output = sequence_output[:, 0, :]  # Take the representation of the [CLS] token
            logits = nsp_classifier_layer(pooled_output)

            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    documents = [
        "The weather is nice today. I plan to go for a walk. There are many flowers in bloom.",
        "Artificial intelligence is a growing field. It has applications in many areas. The future is promising."
    ]

    model_name='roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    dataset = prepare_nsp_dataset(documents, tokenizer)
    print(dataset[:2])  # Show first two examples

    train_nsp_model(dataset, model_name)