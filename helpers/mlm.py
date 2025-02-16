import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, AdamW
from torch import nn

def prepare_mlm_dataset(documents, tokenizer, max_length=128, mask_probability=0.15):
    class MLMDataset(Dataset):
        def __init__(self, tokenized_data):
            self.tokenized_data = tokenized_data

        def __len__(self):
            return len(self.tokenized_data)

        def __getitem__(self, idx):
            return {
                'input_ids': self.tokenized_data[idx]['input_ids'],
                'attention_mask': self.tokenized_data[idx]['attention_mask'],
                'labels': self.tokenized_data[idx]['labels']
            }

    # Function to mask tokens in the input data
    def mask_tokens(inputs, tokenizer, mask_probability=0.15):
        labels = inputs.clone()  # Clone inputs for labels

        # Create a mask for tokens to be replaced
        probability_matrix = torch.full(labels.shape, mask_probability)

        # Get special tokens mask (e.g., [CLS], [SEP], etc.)    
        special_tokens_mask = tokenizer.get_special_tokens_mask(inputs.tolist(), already_has_special_tokens=True)

        # Set probability of masking special tokens to 0
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        # Mask 15% of the tokens
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Set non-masked tokens to -100 (ignored in loss calculation)

        inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        return inputs, labels
        # return torch.tensor(inputs), labels

    def tokenize_and_mask(documents, tokenizer, max_length=128, mask_probability=0.15):
        tokenized_data = []
        for doc in documents:
            # Tokenize the document
            inputs = tokenizer(
                doc,
                return_tensors='pt',
                max_length=max_length,
                padding='max_length',
                truncation=True
            )
            input_ids = inputs['input_ids'].squeeze(0)
            attention_mask = inputs['attention_mask'].squeeze(0)

            # Apply masking
            masked_inputs, labels = mask_tokens(input_ids, tokenizer, mask_probability)

            tokenized_data.append({
                'input_ids': masked_inputs,
                'attention_mask': attention_mask,
                'labels': labels,
            })

        return tokenized_data

    dataset = tokenize_and_mask(documents, tokenizer, max_length=max_length, mask_probability=mask_probability)
    return MLMDataset(dataset)

# class RobertaForCustomMaskedLM(nn.Module):
#     def __init__(self, model_name):
#         super(RobertaForCustomMaskedLM, self).__init__()
#         self.roberta = RobertaModel.from_pretrained(model_name)
#         self.mlm = nn.Linear(self.roberta.config.hidden_size, self.roberta.config.vocab_size)

#     def forward(self, input_ids, labels=None):
#         outputs = self.roberta(input_ids)
#         sequence_output = outputs[0]
#         prediction_scores = self.mlm(sequence_output)

#         loss = None
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             # Shift prediction scores and labels for the loss calculation
#             prediction_scores = prediction_scores.view(-1, self.roberta.config.vocab_size)
#             labels = labels.view(-1)
#             loss = loss_fct(prediction_scores, labels)

#         return {'loss': loss, 'logits': prediction_scores}

def train_mlm_model(dataset, model_name='roberta-base', batch_size=8, num_epochs=30, learning_rate=5e-5):
    # Initialize model
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model = RobertaForCustomMaskedLM(model_name)

    backbone_model = RobertaModel.from_pretrained(model_name)
    mlm_classifier_layer = nn.Linear(backbone_model.config.hidden_size, backbone_model.config.vocab_size)
    optimizer = AdamW(list(backbone_model.parameters()) + list(mlm_classifier_layer.parameters()), lr=learning_rate)
    loss_fct = nn.CrossEntropyLoss()

    # Training loop
    backbone_model.train()
    mlm_classifier_layer.train()
    for epoch in range(num_epochs):  # Number of epochs
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            # outputs = backbone_model(input_ids=batch['input_ids'], labels=batch['labels'])
            # loss = outputs['loss']

            outputs = backbone_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            sequence_output = outputs[0]
            prediction_scores = mlm_classifier_layer(sequence_output)
            prediction_scores = prediction_scores.view(-1, backbone_model.config.vocab_size)
            labels = batch['labels'].view(-1)
            loss = loss_fct(prediction_scores, labels)

            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # Example usage
    documents = [
        "The weather is nice today. I plan to go for a walk. There are many flowers in bloom.",
        "Artificial intelligence is a growing field. It has applications in many areas. The future is promising."
    ]

    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    dataset = prepare_mlm_dataset(documents, tokenizer)
    
    train_mlm_model(dataset, model_name)