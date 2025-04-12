import csv
import json

import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
import pickle

import keras
import os
from keras import layers
from keras import utils
from keras import Model
from keras import models
from keras import optimizers
from keras import callbacks
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
import random

import tensorflow as tf


class GestureDataProcessor:
    def __init__(self, test = 0):
        #self.feature_match = {"fixcenter": 1, "highdynamic": 2, "lowdynamic": 3, "speaker": 4, "relax": 5}
        self.feature_match = {"fashion": 1, "game": 2, "music": 3, "news": 4, "podcast": 5, "movie": 6, "sport":7}
        self.gesture_name = ["fashion", "game", "music", "news", "podcast","movie","sport"]
        self.all_video_files = [1,2,3,4,5,6,7,8,9,10]

        self.test = test
        self.loaded_x = list()
        self.loaded_y = list()
        self.testFile = list()
        self.trainFile = list()
        self.stepLen = '32'
        self.totalAcc = float()

        self.folder_path = "all_gazes_text/youtube_video_processed/"
        #self.data_split()



    # separate self.all_video_files into self.trainFile and self.testFile
    # train_ratio means how many percentage files will be used for train
    def data_split(self, train_ratio=0.8):
        if self.test == 1:
            self.trainFile = [[1,2,3,4,6,7,8,9]]
            self.testFile = [[5,10]]
            print(self.trainFile)
            print(self.testFile)
            return

        self.trainFile = list()
        self.testFile = list()

        shuffled_files = self.all_video_files[:]
        random.shuffle(shuffled_files)

        # Calculate the split index
        split_index = int(len(shuffled_files) * train_ratio)
        # Split the files
        self.trainFile.append(shuffled_files[:split_index])
        print(self.trainFile)
        self.testFile.append(shuffled_files[split_index:])
        print(self.testFile)



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

        #window size means number of seconds does each window has
        window_size = 4
        step = (window_size * int(self.stepLen)) // 2  # 50% overlap

        print("process clips")
        # Split the combined_list into sublists
        for i in range(0, len(combined_list), step):
            time_step_data = combined_list[i:i + (window_size * int(self.stepLen))]
            if len(time_step_data) == (window_size * int(self.stepLen)):
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

    def plot_training_history(self, history):
        plt.figure(figsize = (14, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label = 'Train Loss')
        plt.plot(history.history['val_loss'], label = 'Val Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label = 'Train Accuracy')
        plt.plot(history.history['val_accuracy'], label = 'Val Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # run experiment for once.
    def evaluate_model(self, trainX, trainy, testX_list, testy):

        verbose, epochs, batch_size = 0, 60, 32
        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

        checkpoint = callbacks.ModelCheckpoint(
            "best_model.keras",
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        )

        if self.test == 2:
            print("test saved model")
            model = models.load_model("best_model.keras")

        else:
            print(n_timesteps, n_features, n_outputs)
            print(type(trainX),type(trainy),type(testX_list),type(testy))

            # Build the model using Sequential API
            inputs = layers.Input(shape = (n_timesteps, n_features))

            # 1D CNN block
            conv1 = layers.Conv1D(filters = 256, kernel_size = 7, activation = 'relu', padding = 'same')(inputs)
            conv2 = layers.Conv1D(filters = 256, kernel_size = 7, activation = 'relu', padding = 'same')(conv1)
            pool1 = layers.MaxPooling1D(pool_size = 2)(conv2)
            conv3 = layers.Conv1D(filters = 256, kernel_size = 7, activation = 'relu', padding = 'same')(pool1)
            pool2 = layers.MaxPooling1D(pool_size = 2)(conv3)

            conv4 = layers.Conv1D(filters = 256, kernel_size = 3, activation = 'relu',
                                  padding = 'same')(pool2)
            pool3 = layers.MaxPooling1D(pool_size = 2)(conv4)

            flat = layers.Flatten()(pool3)
            dense_out = layers.Dense(100, activation = 'relu')(flat)
            dense_out = layers.Dropout(0.5)(dense_out)

            # Output layer
            output = layers.Dense(n_outputs, activation = 'softmax')(dense_out)

            # Compile
            model = Model(inputs = inputs, outputs = output)
            #optimizer = optimizers.Adam(learning_rate = 1e-4)
            model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

            trainX_part, valX, trainy_part, valy = train_test_split(
                trainX, trainy, test_size = 0.1, stratify = trainy.argmax(axis = 1), random_state = 42
            )

            # Training
            early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10)
            history = model.fit(trainX_part, trainy_part,
                        validation_data=(valX, valy), epochs = epochs, batch_size = batch_size, verbose = verbose,
                         callbacks=[early_stopping, checkpoint])
            self.plot_training_history(history)
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
            print("Predicted clips:", y_pred_labels)

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
        Ave_acc = 'Average accuracy: %.3f%% (+/-%.3f)' % (m, s)
        print(Ave_acc)

        return Ave_acc

    def to_csv(self, ave_acc, overall_conf_matrix):
        csv_file = 'Model_eval.csv'
        # Convert the confusion matrix to a string to save as one block
        conf_matrix_str = np.array2string(overall_conf_matrix, separator = ',',
                                          formatter = {'int': lambda x: "%d" % x})

        # Load the CSV file
        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            # If file doesn't exist, create one with a column named '64'
            df = pd.DataFrame(columns = [self.stepLen])

        # Find the first empty row in the '64' column
        first_empty_idx = df[self.stepLen].isna().idxmax()

        # If all rows are filled, append at the end of the file
        if first_empty_idx is None:
            print("append at end.")
            first_empty_idx = len(df)

        # Expand DataFrame if necessary
        if len(df) < first_empty_idx + 1:
            print("expand dataframe")
            # Expand DataFrame by adding one additional row
            df = pd.concat(
                [df, pd.DataFrame(np.nan, index = [first_empty_idx], columns = df.columns)],
                ignore_index = True)

        # Insert mean score and confusion matrix as a string into the row
        df.at[first_empty_idx, self.stepLen] = ave_acc
        df.at[first_empty_idx, self.stepLen] += conf_matrix_str

        # Save the updated DataFrame back to the CSV
        df.to_csv(csv_file, index = False)

    # run experiment multiple times.
    # each time dataset is loaded randomly and data is evaluated by model trained by train data.
    def run_experiment(self, repeats = 1):
        # repeat experiment
        scores = list()
        overall_conf_matrix = None

        trainX, trainy, testX, testy = None, None, None, None

        for r in range(repeats):
            if self.test == 0 or self.test == 2:
                trainX, trainy, testX, testy = self.load_dataSet(0)
            elif self.test == 1:
                if os.path.exists('data.pkl'):
                    print(f"Loading data from data.pkl")
                    if trainX is None:
                        with open('data.pkl', 'rb') as f:
                            trainX, trainy, testX, testy = pickle.load(f)
                    else:
                        #already loaded
                        pass
                else:
                    print("data.pkl not found. Saving data.")
                    trainX, trainy, testX, testy = self.load_dataSet(0)
                    with open('data.pkl', 'wb') as f:
                        pickle.dump((trainX, trainy, testX, testy), f)

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
        plt.title(f"Overall Confusion Matrix Heatmap\n{ave_acc}")
        plt.show()

        overall_conf_matrix = overall_conf_matrix / overall_conf_matrix.sum(axis = 1,
                                                                                       keepdims = True) * 100
        # Plot heatmap for the overall confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(overall_conf_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"Overall Confusion Matrix Heatmap percentage\n{ave_acc}")
        plt.show()

        #self.to_csv(ave_acc, overall_conf_matrix)



# Usage example:
processor = GestureDataProcessor(1)

processor.run_experiment(2)
