import csv
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std

import keras
import os
from keras import layers
from keras import utils
from keras import Model
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from collections import Counter
import random

class GestureDataProcessor:
    def __init__(self):
        #self.feature_match = {"fixcenter": 1, "highdynamic": 2, "lowdynamic": 3, "speaker": 4, "relax": 5}
        self.feature_match = {"fashion": 1, "game": 2, "music": 3, "news": 4, "podcast": 5, "sport": 6}
        self.gesture_name = ["fashion", "game", "music", "news", "podcast"]
        self.all_video_files = [1,2,3,4,5,6,7,8,9,10]


        self.loaded_x = list()
        self.loaded_y = list()
        self.testFile = list()
        self.trainFile = list()
        self.stepLen = '32'
        self.totalAcc = float()

        self.folder_path = "all_gazes_text/youtube_video_type/"
        #self.data_split()



    # separate self.all_video_files into self.trainFile and self.testFile
    # train_ratio means how many percentage files will be used for train
    def data_split(self, train_ratio=0.8):
        self.trainFile = list()
        self.testFile = list()

        shuffled_files = self.all_video_files[:]
        random.shuffle(shuffled_files)

        # Calculate the split index
        split_index = int(len(shuffled_files) * train_ratio)
        # Split the files
        self.trainFile.append(shuffled_files[:split_index])
        self.testFile.append(shuffled_files[split_index:])



    def video_to_clips(self, file_path):
        # Initialize lists to store the data
        combined_list = list()
        sublists = list()

        # Read the file and parse the data
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():  # Ensure the line is not empty
                    # Split the line into its components
                    parts = line.split(",")
                    position_x = parts[2].replace("current eyeGaze_left: Position ", "").strip()
                    pos_x = float(position_x.replace("(", ""))
                    pos_y = float(parts[3])
                    pos_z = float(parts[4].replace(")", ""))

                    orientation_x = parts[5].replace("Orientation", "").strip()
                    ori_x = float(orientation_x.replace("(", ""))
                    ori_y = float(parts[6])
                    ori_z = float(parts[7])
                    ori_w = float(parts[8].replace(")", ""))

                    combined_list = combined_list + [ori_x, ori_y, ori_z, ori_w]

        #16 means number of seconds does each step has
        step = (32 * int(self.stepLen)) // 2  # 50% overlap

        # Split the combined_list into sublists
        for i in range(0, len(combined_list), step):
            time_step_data = combined_list[i:i + (16 * int(self.stepLen))]
            if len(time_step_data) == (16 * int(self.stepLen)):
                # sublist is a list which contains many [ori_x, ori_y, ori_z, ori_w], means a clip in 16 seconds.
                sublist = [time_step_data[i:i + 4] for i in range(0, len(time_step_data), 4)]
                # sublists is a list that contains all the clips of this file.
                sublists.append(sublist)

        return sublists

    # load data, train and test files are treated differently
    def load_group(self, gesture_names, group, combination_index):
        if group == "train":
            range_domain = self.trainFile[combination_index]

            for gesture_name in gesture_names:

                for i in range_domain:
                    file_path = os.path.join(self.folder_path, f"{gesture_name}{i}.txt")
                    data_list = self.video_to_clips(file_path)

                    for clip in data_list:
                        self.loaded_y.append([self.feature_match[gesture_name]])

                    self.loaded_x += data_list

            loaded_data_x = np.array(self.loaded_x)
            loaded_data_y = np.array(self.loaded_y)
            # print(loaded_data_x)
            # print(loaded_data_y)
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
                    self.loaded_y.append([self.feature_match[gesture_name]])

                    #make self.loaded_x a list of np array, each np array represent clips in a video
                    self.loaded_x.append(data_list)

            loaded_data_x = self.loaded_x

            print(f"Totally we have {len(loaded_data_x)} videos for testing.")
            #exit(0)
            loaded_data_y = np.array(self.loaded_y)
            self.loaded_y = list()
            self.loaded_x = list()

            return loaded_data_x, loaded_data_y

    # randomly choose train and test files
    # process train and test files and returns them
    def load_dataSet(self, combination_index):
        #initialize the train(80%) and test(20%) data set.
        self.data_split()

        trainX, trainY = self.load_group(self.gesture_name, "train", combination_index)
        #print(trainX.shape, trainY.shape)
        testX, testY = self.load_group(self.gesture_name, "test", combination_index)
        #print(testX.shape, testY.shape)

        # zero-offset class values
        trainY = trainY - 1
        testY = testY - 1
        # one hot encode y
        trainY = utils.to_categorical(trainY)
        testY = utils.to_categorical(testY)
        #print(trainX.shape, trainY.shape, testX.shape, testY.shape)
        return trainX, trainY, testX, testY

    # run experiment for once.
    def evaluate_model(self, trainX, trainy, testX_list, testy):

        verbose, epochs, batch_size = 0, 30, 64
        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

        # Build the model using Sequential API
        inputs = layers.Input(shape = (n_timesteps, n_features))

        # LSTM layer with return_sequences=True to maintain the time steps for attention
        lstm_out = layers.LSTM(100, return_sequences = True)(inputs)

        # Attention mechanism: Apply self-attention (query and value are both LSTM outputs)
        attention_out = layers.Attention()([lstm_out, lstm_out])
        #attention_out = layers.MultiHeadAttention(num_heads=4, key_dim=100)(lstm_out, lstm_out)

        # Flatten the attention output to feed into Dense layers
        flattened_out = layers.Flatten()(attention_out)

        # Add the dense layer with 100 units and ReLU activation (same as original)
        dense_out = layers.Dense(100, activation = 'relu')(flattened_out)

        # Dropout layer for regularization
        dense_out = layers.Dropout(0.3)(dense_out)

        # Final dense layer with softmax activation (same as original)
        output = layers.Dense(n_outputs, activation = 'softmax')(dense_out)

        # Build the model
        model = Model(inputs = inputs, outputs = output)
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        # An epoch refers to one complete pass through the entire training dataset
        # Stop if model doesn’t improve on the validation set for 5 consecutive epochs
        early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)
        model.fit(trainX, trainy, epochs, batch_size, verbose=verbose, validation_split=0.1, callbacks=[early_stopping])

        # Fit the model
        #model.fit(trainX, trainy, epochs = epochs, batch_size = batch_size, verbose = verbose)

        # Evaluate the model on test data
        #_, accuracy = model.evaluate(testX, testy, batch_size = batch_size, verbose = 0)

        # Predict the test data and calculate the confusion matrix
        label_list = list()
        for testX in testX_list:
            y_pred = model.predict(testX, batch_size = batch_size, verbose = 0)
            y_pred_labels = np.argmax(y_pred, axis = 1)
            y_pred_labels = y_pred_labels.tolist()
            #print("Predicted Label:", y_pred_labels)

            # Find the majority number (most common label)
            if y_pred_labels:  # Ensure y_pred_labels is not empty
                majority_label = Counter(y_pred_labels).most_common(1)[0][0]
                label_list.append(majority_label)

        # Convert both to numpy arrays
        predicted_labels = np.array(label_list)
        print("Predicted Labels:", predicted_labels)
        true_labels = np.argmax(testy, axis = 1)
        print("True Labels:", true_labels)

        # results are a list of labels.
        #print("Predicted Labels:", y_pred_labels.tolist())
        #print("True Labels:", y_true_labels.tolist())

        #exit(0)

        # Calculate accuracy
        accuracy = np.mean(predicted_labels == true_labels)
        print("Accuracy:", accuracy)

        # Print the confusion matrix
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        print(conf_matrix / conf_matrix.astype("float").sum(axis=1))

        return accuracy, conf_matrix


    def summarize_results(self, scores):
        #print(scores)
        m, s = mean(scores), std(scores)
        Ave_acc = 'Accuracy: %.3f%% (+/-%.3f)' % (m, s)
        print(Ave_acc)

        return Ave_acc

    # run experiment multiple times.
    # each time dataset is loaded randomly and data is evaluated by model trained by train data.
    def run_experiment(self, repeats = 10):
        # repeat experiment
        scores = list()
        overall_conf_matrix = None

        for r in range(repeats):

            trainX, trainy, testX, testy = self.load_dataSet(0)

            score, conf_matrix = self.evaluate_model(trainX, trainy, testX, testy)
            score = score * 100.0
            print('>#%d: %.3f' % (r + 1, score))
            scores.append(score)

            # Accumulate confusion matrices
            if overall_conf_matrix is None:
                overall_conf_matrix = conf_matrix
            else:
                overall_conf_matrix += conf_matrix

        # summarize results
        ave_acc = self.summarize_results(scores)
        self.totalAcc += mean(scores)
        print(f"Overall Confusion Matrix:\n{overall_conf_matrix}")

        # Plot heatmap for the overall confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(overall_conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=True)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Overall Confusion Matrix Heatmap")
        plt.show()

        overall_conf_matrix = overall_conf_matrix / overall_conf_matrix.sum(axis = 1,
                                                                                       keepdims = True) * 100
        # Plot heatmap for the overall confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(overall_conf_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Overall Confusion Matrix Heatmap percentage")
        plt.show()

        #self.to_csv(ave_acc, overall_conf_matrix)

        print(f"Average acc is {self.totalAcc/len(self.trainFile)}%")



# Usage example:
processor = GestureDataProcessor()

processor.run_experiment()
