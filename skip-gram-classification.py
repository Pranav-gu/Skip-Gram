import torch
import numpy as np
import math
import torch.nn as nn
import pandas as pd
import re

device = 'cuda' if torch.cuda.is_available() else 'cpu'
skip_gram_embeddings = torch.load("skip-gram-word-vectors.pt")
skip_gram_embeddings['OOV'] = torch.randn(300, dtype = torch.float32)

class LSTM(nn.Module):
    def __init__(self, hidden_dim, output_dim, embedding_dim = 300, num_layers = 1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # Initialize hidden state
        output, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Take the output from the last time step

        output_last = output[:, -1]
        # Pass it through the fully connected layer
        output_fc = self.fc(output_last)
        
        return output_fc

        # output, (_, _) = self.lstm(x)
        # return self.fc(output)

data_train = pd.read_csv('train.csv')
length = []
for i in range(len(data_train['Description'])):
    sentence = re.findall(r"[\w']+|[.,!?;'-]", data_train['Description'][i])
    length.append(len(sentence))
length.sort()
max_len = length[int(0.95*len(length))]

data = pd.read_csv('test.csv')

hidden_dim = 256
output_dim = data['Class Index'].unique().shape[0]

model = LSTM(hidden_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
criterion = nn.CrossEntropyLoss()

batch_size = 100
model.train()
num_epochs = 10
for epoch in range(num_epochs):
    correct = 0
    train_loss = 0

    for i in range(int(len(data_train['Description'])/batch_size)):
        optimizer.zero_grad()
        batch_embeddings = torch.zeros(batch_size, max_len, 300, dtype = torch.float32, device = device)

        for k in range(batch_size*i, batch_size*i+batch_size):
            sentence = re.findall(r"[\w']+|[.,!?;'-]", data_train['Description'][k])
            for j, word in enumerate(sentence):
                if (j >= max_len):
                    break
                if (word not in skip_gram_embeddings):
                    batch_embeddings[k-batch_size*i][j] = skip_gram_embeddings['OOV']
                else:
                    batch_embeddings[k-batch_size*i][j] = torch.from_numpy(skip_gram_embeddings[word])

            for j in range(max(0, max_len-len(sentence))):                      # Pad the sentences with the PAD Token or OOV Embeddings
                batch_embeddings[k-batch_size*i][j+len(sentence)] = skip_gram_embeddings['OOV']
        y_pred = model(batch_embeddings).to(device)

        # y_true = torch.Tensor(data_train['Class Index'][100*i:100*(i+1)]).to(torch.int64)
        y_true = torch.tensor(data_train['Class Index'].values[batch_size*i:batch_size*(i+1)], dtype=torch.int64, device=device)
        loss = criterion(y_pred, y_true-1)
        train_loss += loss

        predicted = torch.argmax(y_pred, dim = 1)
        correct += torch.sum(predicted+1 == y_true).item()
        loss.backward()
        optimizer.step()
    print(f"Epoch = {epoch}\tTraining Set Accuracy = {correct / len(data_train['Description'])}\tLoss = {train_loss/len(data_train['Description'])}")


batch_size = 100
model.eval()
correct = 0
test_loss = 0

for i in range(int(len(data['Description'])/batch_size)):
    batch_embeddings = torch.zeros(batch_size, max_len, 300, dtype = torch.float32, device = device)

    for k in range(batch_size*i, batch_size*i+batch_size):
        sentence = re.findall(r"[\w']+|[.,!?;'-]", data['Description'][k])
        for j, word in enumerate(sentence):
            if (j >= max_len):
                break
            if (word not in skip_gram_embeddings):
                batch_embeddings[k-batch_size*i][j] = skip_gram_embeddings['OOV']
            else:
                batch_embeddings[k-batch_size*i][j] = torch.from_numpy(skip_gram_embeddings[word])

        for j in range(max(0, max_len-len(sentence))):                      # Pad the sentences with the PAD Token or OOV Embeddings
            batch_embeddings[k-batch_size*i][j+len(sentence)] = skip_gram_embeddings['OOV']
    y_pred = model(batch_embeddings).to(device)

    # y_true = torch.Tensor(data_train['Class Index'][100*i:100*(i+1)]).to(torch.int64)
    y_true = torch.tensor(data['Class Index'].values[batch_size*i:batch_size*(i+1)], dtype=torch.int64, device=device)
    loss = criterion(y_pred, y_true-1)
    test_loss += loss
    predicted = torch.argmax(y_pred, dim = 1)
    correct += torch.sum(predicted+1 == y_true).item()
print(f"Testing Set Accuracy = {correct / len(data['Description'])}\tLoss = {test_loss/len(data['Description'])}")

from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix
torch.save(model, "skip-gram-classification-model.pt")

batch_size = 100
model.train()
precision = []
f1 = []
recall = []
correct = 0
train_loss = 0
import warnings
warnings.filterwarnings('ignore')


confusion = None
with torch.no_grad():
    for i in range(int(len(data_train['Description'])/batch_size)):
        batch_embeddings = torch.zeros(batch_size, max_len, 300, dtype = torch.float32, device = device)

        for k in range(batch_size*i, batch_size*i+batch_size):
            sentence = re.findall(r"[\w']+|[.,!?;'-]", data_train['Description'][k])
            for j, word in enumerate(sentence):
                if (j >= max_len):
                    break
                if (word not in skip_gram_embeddings):
                    batch_embeddings[k-batch_size*i][j] = skip_gram_embeddings['OOV']
                else:
                    batch_embeddings[k-batch_size*i][j] = torch.from_numpy(skip_gram_embeddings[word])

            for j in range(max(0, max_len-len(sentence))):                      # Pad the sentences with the PAD Token or OOV Embeddings
                batch_embeddings[k-batch_size*i][j+len(sentence)] = skip_gram_embeddings['OOV']
        y_pred = model(batch_embeddings).to(device)

        # y_true = torch.Tensor(data_train['Class Index'][100*i:100*(i+1)]).to(torch.int64)
        y_true = torch.tensor(data_train['Class Index'].values[batch_size*i:batch_size*(i+1)], dtype=torch.int64, device=device)
        loss = criterion(y_pred, y_true-1)
        train_loss += loss

        predicted = torch.argmax(y_pred, dim = 1)
        correct += torch.sum(predicted+1 == y_true).item()

        pred = (predicted+torch.ones_like(predicted)).cpu().numpy()
        true = y_true.cpu().numpy()
        precision.append(precision_score(pred, true, average='weighted'))
        f1.append(f1_score(pred, true, average='weighted'))
        recall.append(recall_score(pred, true, average='weighted'))
        confusion = confusion_matrix(pred, true)
        print(f"Confusion Matrix for Batch = {i} is {confusion}")

    print(f"Training Set Accuracy = {correct / len(data_train['Description'])}\tRecall = {np.mean(recall)}\tPrecision = {np.mean(precision)}\tF1-Score = {np.mean(f1)}\tLoss = {train_loss/len(data_train['Description'])}\tConfusion Matrix = {confusion}")


batch_size = len(data['Description'])
model.eval()
correct = 0
test_loss = 0
precision = []
f1 = []
recall = []


for i in range(int(len(data['Description'])/batch_size)):
    batch_embeddings = torch.zeros(batch_size, max_len, 300, dtype = torch.float32, device = device)

    for k in range(batch_size*i, batch_size*i+batch_size):
        sentence = re.findall(r"[\w']+|[.,!?;'-]", data['Description'][k])
        for j, word in enumerate(sentence):
            if (j >= max_len):
                break
            if (word not in skip_gram_embeddings):
                batch_embeddings[k-batch_size*i][j] = skip_gram_embeddings['OOV']
            else:
                batch_embeddings[k-batch_size*i][j] = torch.from_numpy(skip_gram_embeddings[word])

        for j in range(max(0, max_len-len(sentence))):                      # Pad the sentences with the PAD Token or OOV Embeddings
            batch_embeddings[k-batch_size*i][j+len(sentence)] = skip_gram_embeddings['OOV']
    y_pred = model(batch_embeddings).to(device)

    # y_true = torch.Tensor(data_train['Class Index'][100*i:100*(i+1)]).to(torch.int64)
    y_true = torch.tensor(data['Class Index'].values[batch_size*i:batch_size*(i+1)], dtype=torch.int64, device=device)
    loss = criterion(y_pred, y_true-1)
    test_loss += loss
    predicted = torch.argmax(y_pred, dim = 1)
    correct += torch.sum(predicted+1 == y_true).item()

    pred = (predicted+torch.ones_like(predicted)).cpu().numpy()
    true = y_true.cpu().numpy()
    precision.append(precision_score(pred, true, average='weighted'))
    f1.append(f1_score(pred, true, average='weighted'))
    recall.append(recall_score(pred, true, average='weighted'))
    confusion = confusion_matrix(pred, true)
    print(f"Confusion Matrix for Batch = {i} is {confusion}")

print(f"Testing Set Accuracy = {correct / len(data['Description'])}\tRecall = {np.mean(recall)}\tPrecision = {np.mean(precision)}\tF1-Score = {np.mean(f1)}\tLoss = {test_loss/len(data['Description'])}")