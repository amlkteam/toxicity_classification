# Author: Nilan Saha
# Date: 8th April, 2020


# You have to uncomment the following three lines if you are running this on Colab
!pip install transformers
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import time
import torch
import logging
import transformers
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

model_class = transformers.BertModel
tokenizer_class = transformers.BertTokenizer
pretrained_weights='bert-base-uncased'
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
bert_model = model_class.from_pretrained(pretrained_weights)

logging.getLogger("transformers").setLevel(logging.ERROR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = '/content/drive/My Drive/Colab Notebooks/Toxic Comments/'
df = pd.read_csv(data_path + 'train.csv')
df = df.drop(['id'], axis = 1)
df = df.sample(frac=1, random_state = 42)

toxic = df[df.toxic == 1]
non_toxic = df[df.toxic != 1]
non_toxic = non_toxic.sample(n = 15000)
df = pd.concat([toxic, non_toxic])
df = df.sample(frac=1, random_state = 42)

train, val, test = df[:20000].values, df[20000:25000].values, df[25000:].values
print('Train Size', train.shape)
print('Val Size', val.shape)
print('Test Size', test.shape)

class ToxicDataset(Dataset):
    def __init__(self, dataframe, max_len):
        self.dataframe = dataframe
        self.max_len = max_len
        self.sep_id = tokenizer.encode(['[SEP]'], add_special_tokens=False)
        self.pad_id = tokenizer.encode(['[PAD]'], add_special_tokens=False)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe[idx]
        text = row[0]
        targets = torch.tensor(list(row[1:]))
        encoded = tokenizer.encode(text, add_special_tokens=True)[:self.max_len-1]
        if encoded[-1] != self.sep_id[0]:
            encoded = encoded + self.sep_id
        padded = encoded + self.pad_id * (self.max_len - len(encoded))
        padded = torch.tensor(padded)
        labels = torch.Tensor(list(row[1:]))
        return padded, labels


train_dataset = ToxicDataset(train, 84)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

val_dataset = ToxicDataset(val, 84)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)

test_dataset = ToxicDataset(test, 84)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

def evaluate(model, data):
    actual, predictions = [], []
    with torch.no_grad():
        for features, targets in data:
            features = features.to(device)
            targets = targets.to(device)
            scores, attentions = model(features)
            sigmoid_out = torch.sigmoid(scores)
            prediction = torch.as_tensor(sigmoid_out > 0.5, dtype=torch.int32)
            predictions.extend(prediction.view(-1).tolist())
            actual.extend(targets.long().view(-1).tolist())
    assert len(actual) == len(predictions)
    print('Macro F1 Score', f1_score(actual, predictions, average = 'macro'))

class BertNN(nn.Module):
    def __init__(self, hidden_size):
        super(BertNN, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained(pretrained_weights, output_attentions = True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, 6)

    def forward(self, ex):
        _, pooled_output, attentions = self.bert_model(ex)
        pooled_output = self.dropout(pooled_output)
        fc_out = self.fc(pooled_output)
        return fc_out, attentions


model = BertNN(768)
model = model.to(device)
loss_function = nn.BCEWithLogitsLoss()
loss_function = loss_function.to(device)
optimizer = transformers.AdamW(model.parameters(), lr=2e-5, correct_bias=False)
MAX_EPOCHS = 1

max_grad_norm = 1.0
warmup_proportion = 0.1
num_training_steps  = len(train_dataloader) * MAX_EPOCHS
num_warmup_steps = num_training_steps * warmup_proportion
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

for epoch in range(MAX_EPOCHS):
    epoch_loss = 0
    start_time = time.time()
    for idx, (features, targets) in enumerate(train_dataloader):
        model.zero_grad()
        features = features.to(device)
        targets = targets.to(device)
        scores, attentions = model(features)
        loss = loss_function(scores, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
    if device == 'cuda':
        torch.cuda.empty_cache()
    time_taken = round((time.time() - start_time)/60, 2)
    print(f'Epoch {epoch + 1} | Loss - {epoch_loss} | Time Taken - {time_taken} min')
    evaluate(model, val_dataloader)

evaluate(model, test_dataloader)
