import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Imports ---
import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
class HARLSTM(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=100, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, n_outputs)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransferLSTMModel(nn.Module):
    def __init__(self, pretrained_model, input_size, hidden_size, output_size):
        super(TransferLSTMModel, self).__init__()
        self.input_proj = nn.LSTM(input_size=input_size, hidden_size=50, batch_first=True)
        self.transfer_fc = nn.Linear(50, hidden_size)
        self.pretrained_layers = nn.Sequential(
            pretrained_model.dropout,
            pretrained_model.fc1,
        )
        self.output = nn.Linear(100, output_size)

        for param in self.pretrained_layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        x, _ = self.input_proj(x)
        x = x[:, -1, :]
        x = F.relu(self.transfer_fc(x))
        x = self.pretrained_layers(x)
        x = self.output(x)
        return x

import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
from collections import Counter

def train_transfer_model(model, trainX, trainY, device='cpu', epochs=30, batch_size=128):
    dataset = TensorDataset(torch.tensor(trainX, dtype=torch.float32),
                            torch.tensor(np.argmax(trainY, axis=1), dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

def evaluate_transfer_model(model, testX_list, testY, device='cpu'):
    model.eval()
    label_list = []
    with torch.no_grad():
        for testX in testX_list:
            test_tensor = torch.tensor(testX, dtype=torch.float32).to(device)
            out = model(test_tensor)
            preds = out.argmax(dim=1).cpu().numpy().tolist()
            if preds:
                majority = Counter(preds).most_common(1)[0][0]
                label_list.append(majority)

    predicted_labels = np.array(label_list)
    true_labels = np.argmax(testY, axis=1)
    acc = np.mean(predicted_labels == true_labels)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("✅ Accuracy:", acc)
    print("Confusion matrix (%):\n", conf_matrix / conf_matrix.sum(axis=1, keepdims=True))
    return acc, conf_matrix

class GestureDataProcessor:
    def __init__(self, test=0):
        self.feature_match = {"fashion": 1, "game": 2, "music": 3, "news": 4, "podcast": 5, "movie": 6, "sport": 7}
        self.gesture_name = list(self.feature_match.keys())
        self.all_video_files = list(range(1, 11))
        self.test = test
        self.loaded_x = []
        self.loaded_y = []
        self.testFile = []
        self.trainFile = []
        self.stepLen = '32'
        self.folder_path = "all_gazes_text/youtube_video_processed/"
        self.combinations = [
            ([1, 2, 3, 6, 7, 8, 9, 10], [4, 5]),
            ([1, 2, 4, 5, 6, 8, 9, 10], [3, 7]),
            ([1, 3, 5, 6, 7, 8, 9, 10], [2, 4]),
            ([2, 3, 4, 5, 6, 7, 8, 10], [1, 9]),
            ([1, 2, 3, 4, 5, 6, 7, 9], [8, 10]),
            ([1, 2, 3, 4, 6, 7, 8, 9], [5, 10]),
            ([1, 2, 3, 5, 6, 7, 8, 10], [4, 9]),
            ([1, 2, 4, 5, 6, 7, 9, 10], [3, 8]),
            ([1, 3, 4, 5, 6, 8, 9, 10], [2, 7]),
            ([2, 3, 4, 5, 7, 8, 9, 10], [1, 6])

        ]
        self.experiment_count = 0

    def data_random(self, train_ratio=0.8):
        print("random dataset.")
        shuffled = self.all_video_files[:]
        random.shuffle(shuffled)
        split = int(len(shuffled) * train_ratio)
        self.trainFile = [shuffled[:split]]
        self.testFile = [shuffled[split:]]

    def data_split(self):
        if self.experiment_count >= len(self.combinations):
            raise IndexError("No more predefined combinations available.")

        train, test = self.combinations[self.experiment_count]
        self.trainFile = [train]
        self.testFile = [test]
        print(f"🔁 Experiment {self.experiment_count + 1}: Train {train} | Test {test}")
        self.experiment_count += 1



    def video_to_clips(self, path):
        combined = []
        sublists = []

        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.split(",")
                    ori_x = float(parts[5].replace("Orientation", "").strip().replace("(", ""))
                    ori_y = float(parts[6])
                    ori_z = float(parts[7])
                    ori_w = float(parts[8].replace(")", ""))
                    combined += [ori_x, ori_y, ori_z, ori_w]

        window_size = 4
        step = (window_size * int(self.stepLen)) // 2
        for i in range(0, len(combined), step):
            chunk = combined[i:i + (window_size * int(self.stepLen))]
            if len(chunk) == (window_size * int(self.stepLen)):
                sublist = [chunk[i:i + 4] for i in range(0, len(chunk), 4)]
                sublists.append(sublist)

        return sublists

    def load_group(self, gesture_names, group, combination_index):
        if group == "train":
            range_domain = self.trainFile[combination_index]
            data_pairs = []

            for gesture_name in gesture_names:
                for i in range_domain:
                    file_path = os.path.join(self.folder_path, f"{gesture_name}{i}.txt")
                    data_list = self.video_to_clips(file_path)

                    for clip in data_list:
                        #self.loaded_y.append(self.feature_match[gesture_name])
                        label = self.feature_match[gesture_name]
                        data_pairs.append((clip, label))


                    #self.loaded_x += data_list

            # ✅ Shuffle the (clip, label) pairs at the data level
            random.shuffle(data_pairs)
            self.loaded_x, self.loaded_y = zip(*data_pairs)

            loaded_data_x = np.array(self.loaded_x)
            loaded_data_y = np.array(self.loaded_y)
            self.loaded_y = list()
            self.loaded_x = list()

            return loaded_data_x, loaded_data_y

        elif group == "test":
            range_domain = self.testFile[combination_index]

            for gesture_name in gesture_names:

                for i in range_domain:
                    file_path = os.path.join(self.folder_path, f"{gesture_name}{i}.txt")
                    #print(f"{gesture_name}{i}.txt")
                    data_list = self.video_to_clips(file_path)
                    data_list = np.array(data_list)

                    #one label for each video
                    self.loaded_y.append(self.feature_match[gesture_name])

                    #make self.loaded_x a list of np array, each np array represent clips in a video
                    self.loaded_x.append(data_list)

            loaded_data_x = self.loaded_x

            print(f"Totally we have {len(loaded_data_x)} videos for testing.")
            print(self.loaded_y)
            #exit(0)
            loaded_data_y = np.array(self.loaded_y)
            self.loaded_y = list()
            self.loaded_x = list()

            return loaded_data_x, loaded_data_y

    def load_dataset(self, combo_idx=0):
        if self.test == 0:
            self.data_random()
        elif self.test == 1:
            self.data_split()
        else:
            print("Error, undefined test mode, only 1 and 0 are allowed.")
        trainX, trainY = self.load_group(self.gesture_name, "train", combo_idx)
        testX, testY = self.load_group(self.gesture_name, "test", combo_idx)
        trainY, testY = trainY - 1, testY - 1
        return trainX, trainY, testX, testY



    def run_all_combinations(self):
        total_scores = []
        total_conf_matrix = None

        for _ in range(len(self.combinations)):
            trainX, trainY, testX, testY = self.load_dataset()
            print("Train class counts:", np.bincount(trainY))
            print("Test  class counts:", np.bincount(testY))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            score, conf_matrix = train_and_evaluate_model(trainX, trainY, testX, testY, device)
            total_scores.append(score * 100.0)

            if total_conf_matrix is None:
                total_conf_matrix = conf_matrix
            else:
                total_conf_matrix += conf_matrix

        # Final summary
        avg = np.mean(total_scores)
        std = np.std(total_scores)
        print(f"✅ Final Average Accuracy: {avg:.2f}% (+/- {std:.2f}%)")

        # Plot total confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(total_conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title("Overall Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        total_conf_matrix = total_conf_matrix / total_conf_matrix.sum(axis=1, keepdims=True) * 100
        # Plot heatmap for the overall confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(total_conf_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"Overall Confusion Matrix Heatmap percentage\n{avg:.2f}")
        plt.show()

# Step 1: Load pretrained model
pretrained = HARLSTM(n_timesteps=128, n_features=9, n_outputs=6)
pretrained.load_state_dict(torch.load("trained_lstm_model.pth", map_location="cpu"))

# Step 2: Prepare transfer model for gaze-based input
transfer_model = TransferLSTMModel(pretrained, input_size=4, hidden_size=100, output_size=7)
transfer_model.to("cpu")

processor = GestureDataProcessor(test=1)
total_scores = []
total_conf_matrix = None

for _ in range(10):
    trainX, trainY, testX, testY = processor.load_dataset()
    print("Train class counts:", np.bincount(trainY))
    print("Test  class counts:", np.bincount(testY))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Step 4: Train the model on your gaze dataset
    train_transfer_model(transfer_model, trainX, trainY)
    # Step 5: Evaluate on test videos (each testX is a list of clips)
    score, conf_matrix = evaluate_transfer_model(transfer_model, testX, testY)

    total_scores.append(score * 100.0)

    if total_conf_matrix is None:
        total_conf_matrix = conf_matrix
    else:
        total_conf_matrix += conf_matrix

# Final summary
avg = np.mean(total_scores)
std = np.std(total_scores)
print(f"✅ Final Average Accuracy: {avg:.2f}% (+/- {std:.2f}%)")

# Plot total confusion matrix
plt.figure(figsize = (10, 7))
sns.heatmap(total_conf_matrix, annot = True, fmt = 'd', cmap = 'Blues')
plt.title("Overall Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

total_conf_matrix = total_conf_matrix / total_conf_matrix.sum(axis = 1, keepdims = True) * 100
# Plot heatmap for the overall confusion matrix
plt.figure(figsize = (10, 7))
sns.heatmap(total_conf_matrix, annot = True, fmt = ".2f", cmap = "Blues", cbar = True)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title(f"Overall Confusion Matrix Heatmap percentage\n{avg:.2f}")
plt.show()


