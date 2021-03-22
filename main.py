from pathlib import Path

# model_name = "bert-base-uncased"  # This is a big model. Use other models if your computer have memory problems.
model_name = "prajjwal1/bert-tiny"  # This is a big model. Use other models if your computer have memory problems.
# prajjwal1/bert-tiny (L=2, H=128)
# prajjwal1/bert-mini (L=4, H=256)
# prajjwal1/bert-small (L=4, H=512)
# prajjwal1/bert-medium (L=8, H=512)
corpus_name = "AIMed"  # or BioInfer

# Training parameters
learning_rate = 5e-5
num_epochs = 10


import os
def read_dataset(file_name):
    texts = []
    labels = []
    with open(os.path.join("processed_corpus", file_name), "r") as input_file:
        for line in input_file:
            segments = line.strip().split('\t')
            if len(segments) < 3:
                continue
            labels.append(1 if segments[1] == 'P' else 0)
            texts.append(segments[2])
    print(len(texts), len(labels))
    return texts, labels

texts, labels = read_dataset(corpus_name + ".tsv")

from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=.1)
train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=.1)

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)

import torch

class PPIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = PPIDataset(train_encodings, train_labels)
val_dataset = PPIDataset(val_encodings, val_labels)
test_dataset = PPIDataset(test_encodings, test_labels)

from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

optim = AdamW(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()

    model.eval()
    total_loss = 0
    pred_labels = []
    from  sklearn import metrics

    for batch in val_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss = total_loss + outputs.loss
            logits = outputs.logits
            _, this_pred_labels = torch.max(logits, dim=1)
            this_pred_labels = this_pred_labels.cpu()   
            pred_labels = pred_labels + list(this_pred_labels)


    avg_loss = total_loss / len(pred_labels)
    accuracy = metrics.accuracy_score(val_labels, pred_labels)
    precision = metrics.precision_score(val_labels, pred_labels)
    recall = metrics.recall_score(val_labels, pred_labels)
    f1 = metrics.f1_score(val_labels, pred_labels)
    print("Epoch: {}, Accuracy: {:0.2f}, precision: {:0.2f}, recall: {:0.2f}, f1: {:0.2f}".format(epoch, accuracy, precision, recall, f1))