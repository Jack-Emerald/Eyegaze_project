import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import accuracy_score
from numpy import dstack, mean, std
from os.path import join
import os


# ---------- Dataset Loading Functions ----------
def load_file(filepath):
    dataframe = pd.read_csv(filepath, header = None, delim_whitespace = True)
    return dataframe.values


def load_group(filenames, prefix = ''):
    loaded = [load_file(join(prefix, name)) for name in filenames]
    return dstack(loaded)  # shape: (samples, timesteps, features)


def load_dataset_group(group, prefix = ''):
    filepath = join(prefix, group, 'Inertial Signals')
    filenames = [
        f'{sensor}_{axis}_{group}.txt'
        for sensor in ['total_acc', 'body_acc', 'body_gyro']
        for axis in ['x', 'y', 'z']
    ]
    X = load_group(filenames, filepath)
    y = load_file(join(prefix, group, f'y_{group}.txt'))
    return X, y


def load_dataset(prefix = 'HARDataset/'):
    trainX, trainy = load_dataset_group('train', prefix)
    testX, testy = load_dataset_group('test', prefix)

    # zero-based labels
    trainy = trainy.astype(int) - 1
    testy = testy.astype(int) - 1

    print("Train X:", trainX.shape, "Train y:", trainy.shape)
    print("Test  X:", testX.shape, "Test  y:", testy.shape)

    return trainX, trainy.squeeze(), testX, testy.squeeze()


# ---------- PyTorch Dataset ----------
class HARDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# ---------- LSTM Model ----------
class HARLSTM(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super().__init__()
        self.lstm = nn.LSTM(input_size = n_features, hidden_size = 100, batch_first = True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, n_outputs)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq, 100)
        x = lstm_out[:, -1, :]  # take last time step
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------- Train & Evaluate ----------
def evaluate_and_save_model(trainX, trainy, testX, testy, model_filename = 'lstm_model.pth',
                            epochs = 15, batch_size = 64, device = 'cpu'):
    n_timesteps, n_features = trainX.shape[1], trainX.shape[2]
    n_outputs = len(np.unique(trainy))

    model = HARLSTM(n_timesteps, n_features, n_outputs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(HARDataset(trainX, trainy), batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(HARDataset(testX, testy), batch_size = batch_size)

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    preds = []
    truths = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            out = model(x_batch)
            preds.extend(out.argmax(dim = 1).cpu().numpy())
            truths.extend(y_batch.numpy())

    accuracy = accuracy_score(truths, preds)
    torch.save(model.state_dict(), model_filename)
    print(f"✅ Model saved to {model_filename} | Accuracy: {accuracy:.4f}")
    return accuracy


# ---------- Summary ----------
def summarize_results(scores):
    print(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (mean(scores), std(scores)))


# ---------- Run Experiment ----------
def run_experiment(repeats = 1, model_filename = 'trained_lstm_model.pth', use_cuda = True):
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print("🔥 Using device:", device)

    trainX, trainy, testX, testy = load_dataset()
    scores = []
    for r in range(repeats):
        acc = evaluate_and_save_model(trainX, trainy, testX, testy, model_filename, device = device)
        print(f">#{r + 1}: {acc * 100:.2f}%")
        scores.append(acc * 100)
    summarize_results(scores)


# Run it
run_experiment()
