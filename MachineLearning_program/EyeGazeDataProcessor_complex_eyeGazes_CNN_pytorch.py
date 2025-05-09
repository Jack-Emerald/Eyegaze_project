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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Model ---
class GestureCNN(nn.Module):
    def __init__(self, input_len, input_dim, num_classes):
        super(GestureCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=7, padding=3)
        self.pool2 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)
        self._to_linear = self._get_flatten_size(input_len, input_dim)
        self.fc1 = nn.Linear(self._to_linear, 100)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(100, num_classes)

    def _get_flatten_size(self, input_len, input_dim):
        x = torch.zeros(1, input_dim, input_len)
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool3(F.relu(self.conv4(x)))
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool3(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.out(x)


# --- Dataset ---
class GestureDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = torch.tensor(data_x, dtype=torch.float32)
        self.data_y = torch.tensor(data_y.squeeze(), dtype=torch.long)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = self.data_x[idx].permute(1, 0)  # (T, D) -> (D, T)
        y = self.data_y[idx]

        if idx == 0:  # only print for the first sample to avoid spam
            print("🔍 Sample input shape:", x.shape)

        return x, y


# --- Trainer ---
def train_and_evaluate_model(trainX, trainy, testX_list, testy, device, epochs=60, batch_size=48):
    set_seed()

    n_timesteps, n_features = trainX.shape[1], trainX.shape[2]
    n_outputs = len(np.unique(trainy))

    dataset = GestureDataset(trainX, trainy)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # set early stop
    best_val_loss = float('inf')
    patience = 5  # You can tweak this
    patience_counter = 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = GestureCNN(n_timesteps, n_features, n_outputs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for x_batch, y_batch in train_loader:
        print("First batch X:", x_batch.shape)
        print("First batch Y:", y_batch)
        break

    clip_acc = 0

    for epoch in range(epochs):
        model.train()
        train_loss, correct_train = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct_train += (out.argmax(dim=1) == y).sum().item()

        model.eval()
        val_loss, correct_val = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                correct_val += (out.argmax(dim=1) == y).sum().item()

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {correct_train}/{train_size} = {correct_train/train_size:.4f} | Val Acc:{correct_val}/{val_size} = {correct_val/val_size:.4f}")
        if clip_acc < (correct_val/val_size):
            clip_acc = correct_val/val_size

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # Save best model
        else:
            patience_counter += 1
            print(f"⏳ No improvement in val_loss for {patience_counter} epoch(s)")

        if patience_counter >= patience:
            print("🛑 Early stopping triggered")
            break

    # Restore best model
    model.load_state_dict(best_model_state)

    # --- Predict test ---
    label_list = []
    model.eval()
    with torch.no_grad():
        for testX in testX_list:
            testX = torch.tensor(testX, dtype=torch.float32).permute(0, 2, 1).to(device)
            out = model(testX)
            pred = out.argmax(dim=1).cpu().numpy().tolist()
            #print("Predicted clips:", pred)
            if pred:
                majority = Counter(pred).most_common(1)[0][0]
                label_list.append(majority)

    predicted_labels = np.array(label_list)
    print("Predicted Labels:", predicted_labels)
    print("True Labels:", testy.squeeze())
    accuracy = np.mean(predicted_labels == testy.squeeze())
    conf_matrix = confusion_matrix(testy.squeeze(), predicted_labels)
    print(f"✅ Final Test Accuracy: {accuracy:.4f}")
    return accuracy, conf_matrix, clip_acc

class GestureDataProcessor:
    def __init__(self, test=0):
        self.feature_match = {"fashion": 1, "game": 2, "music": 3, "news": 4, "movie": 5, "sport": 6} #"podcast": 5,
        self.gesture_name = list(self.feature_match.keys())
        self.all_video_files = list(range(1, 11))
        self.test = test
        self.loaded_x = []
        self.loaded_y = []
        self.testFile = []
        self.trainFile = []
        self.stepLen = '32'
        self.clip_length = 32
        self.folder_path = "all_gazes_text/youtube_video_processed/"
        self.combinations = [
             ([1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19], [5, 6, 15, 20]),
             ([1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 16, 17, 18, 19, 20], [5, 10, 14, 15]),
             ([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 17, 18, 20], [10, 14, 16, 19]),
             ([1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 17, 18, 20], [7, 13, 16, 19]),
             ([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 17, 19, 20], [7, 11, 13, 18]),
             ([1, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 19, 20], [2, 11, 12, 18]),
             ([1, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20], [2, 3, 12, 17]),
             ([1, 2, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20], [3, 8, 9, 17]),
             ([2, 3, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [1, 4, 8, 9]),
             ([2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 18, 19, 20], [1, 10, 12, 17]),
        ]
        self.experiment_count = 0
        self.test_video_length = 5

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

        window_size = self.clip_length
        step = (window_size * int(self.stepLen)) // 2
        for i in range(0, len(combined), step):
            chunk = combined[i:i + (window_size * int(self.stepLen))]
            if len(chunk) == (window_size * int(self.stepLen)):
                sublist = [chunk[i:i + 4] for i in range(0, len(chunk), 4)]
                sublists.append(sublist)

        return sublists

    def divide_list(self, lst, x):
        k, m = divmod(len(lst), x)
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(x)]

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
                    data_list = self.video_to_clips(file_path) #clips of a video

                    divided = self.divide_list(data_list, self.test_video_length)
                    print(len(divided))

                    for video in divided:
                        data_list = np.array(video)
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

    def run_experiment(self):
        '''
        if os.path.exists('data.pkl'):
            with open('data.pkl', 'rb') as f:
                trainX, trainY, testX, testY = pickle.load(f)
        else:
        '''
        trainX, trainY, testX, testY = self.load_dataset()


        #with open('data.pkl', 'wb') as f:
            #pickle.dump((trainX, trainY, testX, testY), f)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        acc, conf_matrix = train_and_evaluate_model(trainX, trainY, testX, testY, device)

        print(f"✅ Final Test Accuracy: {acc:.4f}")


        '''
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True) * 100
        # Plot heatmap for the overall confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"Overall Confusion Matrix Heatmap percentage\n{acc:.4f}")
        plt.show()
        '''

    def run_all_combinations(self):
        total_scores = []
        total_clip_scores = []
        total_conf_matrix = None

        for _ in range(len(self.combinations)):
            trainX, trainY, testX, testY = self.load_dataset()
            print("Train class counts:", np.bincount(trainY))
            print("Test  class counts:", np.bincount(testY))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            score, conf_matrix, clip_acc = train_and_evaluate_model(trainX, trainY, testX, testY, device)
            total_scores.append(score * 100.0)
            total_clip_scores.append(clip_acc * 100.0)

            if total_conf_matrix is None:
                total_conf_matrix = conf_matrix
            else:
                total_conf_matrix += conf_matrix

        # Final summary
        avg = np.mean(total_scores)
        std = np.std(total_scores)
        avg1 = np.mean(total_clip_scores)
        std1 = np.std(total_clip_scores)
        print(f"✅ Final Average Accuracy: {avg:.2f}% (+/- {std:.2f}%)")
        print(f"✅ Final Average clip Accuracy: {avg1:.2f}% (+/- {std1:.2f}%)")

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


# --- Run the program ---
if __name__ == '__main__':
    processor = GestureDataProcessor(test=1)
    processor.run_all_combinations()