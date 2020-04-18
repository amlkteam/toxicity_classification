# Author: Nilan Saha
# Date: 17th April, 2020


# You have to uncomment the following three lines if you are running this on Colab
#!pip install transformers
#from google.colab import drive
#drive.mount('/content/drive', force_remount=True)

import time
import torch
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader

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

words_set = set()
words_set.add('<PAD>')
words_set.add('<UNK>')

for ex in train:
    text = ex[0]
    tokens = word_tokenize(text)
    for token in tokens:
        words_set.add(token)

print('Word Set Built')

word2idx = {}
for idx, word in enumerate(words_set):
    word2idx[word] = idx

print('Word Index built')

class ToxicDataset(Dataset):
    def __init__(self, dataframe, max_len):
        self.dataframe = dataframe
        self.max_len = max_len
        self.unk_id = word2idx['<UNK>']
        self.pad_id = [word2idx['<PAD>']]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe[idx]
        text = row[0]
        targets = torch.tensor(list(row[1:]))
        encoded = []
        for token in word_tokenize(text):
            encoded.append(word2idx.get(token, self.unk_id))
        encoded = encoded[:self.max_len]
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
            scores = model(features)
            sigmoid_out = torch.sigmoid(scores)
            prediction = torch.as_tensor(sigmoid_out > 0.5, dtype=torch.int32)
            predictions.extend(prediction.view(-1).tolist())
            actual.extend(targets.long().view(-1).tolist())
    assert len(actual) == len(predictions)
    print('Macro F1 Score', f1_score(actual, predictions, average = 'macro'))

class LSTMClassifier(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(len(word2idx), embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 6)
    
    def forward(self, ex):
        embedded = self.embedding(ex)
        embedded = embedded.permute(1,0,2)
        out, _ = self.lstm(embedded)
        out = out[-1]
        fc_out = self.fc(out)
        return fc_out


EMBED_SIZE = 100
HIDDEN_SIZE = 150

model = LSTMClassifier(EMBED_SIZE, HIDDEN_SIZE)
model = model.to(device)
loss_function = nn.BCEWithLogitsLoss()
loss_function = loss_function.to(device)
optimizer = torch.optim.Adam(model.parameters())
MAX_EPOCHS = 20

for epoch in range(MAX_EPOCHS):
    epoch_loss = 0
    start_time = time.time()
    for idx, (features, targets) in enumerate(train_dataloader):
        model.zero_grad()
        features = features.to(device)
        targets = targets.to(device)
        scores  = model(features)
        loss = loss_function(scores, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if device == 'cuda':
        torch.cuda.empty_cache()
    time_taken = round((time.time() - start_time)/60, 2)
    print(f'Epoch {epoch + 1} | Loss - {epoch_loss} | Time Taken - {time_taken} min')
    evaluate(model, val_dataloader)

evaluate(model, test_dataloader)

