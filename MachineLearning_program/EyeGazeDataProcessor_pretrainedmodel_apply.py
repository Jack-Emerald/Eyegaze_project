import csv
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from keras import Sequential
import keras
import os
from keras import layers
from keras import utils
from keras import Model
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix
import random


class GestureDataProcessor:
    def __init__(self, test = 1):
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
        window_size = 16
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
            if self.test == 1:
                loaded_data_x = np.load("train_x.npy")
                loaded_data_y = np.load("train_y.npy")
                return loaded_data_x, loaded_data_y

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
            if self.test == 1:
                loaded_data_x = np.load("test_x.npy")
                loaded_data_y = np.load("test_y.npy")
                return loaded_data_x, loaded_data_y

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

        print(n_timesteps, n_features, n_outputs)

        # Build the model using Sequential API
        inputs = layers.Input(shape = (n_timesteps, n_features))

        # LSTM layer with return_sequences=True to maintain the time steps for attention
        lstm_out = layers.LSTM(100, return_sequences = False)(inputs)

        # Attention mechanism: Apply self-attention (query and value are both LSTM outputs)
        #attention_out = layers.Attention()([lstm_out, lstm_out])
        #attention_out = layers.MultiHeadAttention(num_heads=4, key_dim=100)(lstm_out, lstm_out)

        # Flatten the attention output to feed into Dense layers
        #flattened_out = layers.Flatten()(attention_out)

        # Add the dense layer with 100 units and ReLU activation (same as original)
        dense_out = layers.Dense(100, activation = 'relu')(lstm_out)

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
        Ave_acc = 'Accuracy: %.3f%% (+/-%.3f)' % (m, s)
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

        print(f"Average acc is {self.totalAcc/len(self.trainFile)}%")



class TransferLearningGestureProcessor(GestureDataProcessor):
    def __init__(self, pre_trained_model_path):
        super().__init__()
        self.pre_trained_model_path = pre_trained_model_path

    def evaluate_and_save_model_with_unfreeze(self, trainX, trainy, testX, testy,
                                              model_filename = 'lstm_model.h5'):
        # Load the pre-trained model
        pre_trained_model = models.load_model(self.pre_trained_model_path)

        # Modify the input layer for the new feature size (n_features)
        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

        # Define a new input layer
        inputs = layers.Input(shape = (n_timesteps, n_features))

        # Add an LSTM layer to adapt the input shape (n_features -> 100 units)
        x = layers.LSTM(50, return_sequences = False)(inputs)
        x = layers.Dense(100, activation = 'relu')(x)

        # Pass through the pre-trained LSTM layers (excluding input and output layers)
        pre_trained_layers = pre_trained_model.layers[1:-1]  # Exclude input and output layers
        for layer in pre_trained_layers:
            layer.trainable = False  # Freeze pre-trained layers
            x = layer(x)

        # Add a new output layer for the current task
        output = layers.Dense(n_outputs, activation = 'softmax')(x)

        # Debug: Verify inputs and outputs
        print(f"Input layer: {inputs}")
        print(f"Output layer: {output}")

        # Create the updated model
        transfer_model = models.Model(inputs = inputs, outputs = output)

        # Compile the model
        transfer_model.compile(
            loss = 'categorical_crossentropy',
            optimizer = optimizers.Adam(learning_rate = 0.001),
            # Lower learning rate for fine-tuning
            metrics = ['accuracy']
        )

        # Train the new model on the smaller dataset
        verbose, epochs, batch_size = 0, 60,128

        early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)
        transfer_model.fit(trainX, trainy, epochs = epochs, batch_size = batch_size,
                           verbose = verbose, validation_split = 0.1,
                           callbacks = [early_stopping])

        # Evaluate the model
        #print("Begin evaluate.")
        #_, accuracy = transfer_model.evaluate(testX, testy, batch_size = batch_size, verbose = 0)
        #print(f"Transfer Learning Model Accuracy: {accuracy * 100:.2f}%")

        return transfer_model

    def transfer_learn_model(self, trainX, trainy, testX, testy):
        # Load the pre-trained model
        pre_trained_model = models.load_model(self.pre_trained_model_path)

        # Modify the input layer for the new feature size (4 features)
        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

        # Define a new input layer
        inputs = layers.Input(shape = (n_timesteps, n_features))

        # Add an LSTM layer to adapt the input shape (4 features -> 100 units)
        # Do so since The pre-trained model uses an LSTM layer with 100 units to process sequential data.
        # So pretrained model expect a sequence as input.
        # new input layer is (100,4)
        x = layers.LSTM(100, return_sequences = True)(inputs)

        # Pass through the pre-trained LSTM layers (excluding input and output layers)
        pre_trained_layers = pre_trained_model.layers[1:-1]
        for layer in pre_trained_layers:
            layer.trainable = False
            x = layer(x)

        # Reduce the sequence to a single timestep representation
        # need for next dense layer, which require fixed size vector
        #
        x = layers.LSTM(100, return_sequences = False)(x)

        # Add a new output layer for the 5 labels
        # it's a result of several layers of processing
        output = layers.Dense(n_outputs, activation = 'softmax')(x)

        # Debug: Verify inputs and outputs
        print(f"Input layer: {inputs}")
        print(f"Output layer: {output}")

        # Create a new model
        transfer_model = models.Model(inputs = inputs, outputs = output)

        # Compile the model
        transfer_model.compile(loss = 'categorical_crossentropy',
                               optimizer = optimizers.Adam(learning_rate = 0.001),
                               metrics = ['accuracy'])

        # Train the new model on the smaller dataset
        verbose, epochs, batch_size = 0, 30, 128
        transfer_model.fit(trainX, trainy, epochs = epochs, batch_size = batch_size,
                           verbose = verbose)

        # Evaluate the model
        print("Begin evaluate.")
        _, accuracy = transfer_model.evaluate(testX, testy, batch_size = batch_size, verbose = verbose)
        print(f"Transfer Learning Model Accuracy: {accuracy * 100:.2f}%")

        return transfer_model


# Usage example:
processor = TransferLearningGestureProcessor(pre_trained_model_path='trained_lstm_model.h5')

# Initialize a list to store accuracies
accuracies = []
# Load the dataset (you can vary combination_index if needed)
trainX, trainY, testX, testY = processor.load_dataSet(combination_index = 0)
# Perform the experiment 10 times

scores = list()
overall_conf_matrix = None
for i in range(5):
    print(f"Experiment {i + 1}:")

    # Train the transfer learning model
    #transfer_model = processor.transfer_learn_model(trainX, trainY, testX, testY)
    transfer_model = processor.evaluate_and_save_model_with_unfreeze(trainX, trainY, testX, testY)
    print("done fit model")
    # Predict the test data and calculate the confusion matrix
    label_list = list()
    for X in testX:
        y_pred = transfer_model.predict(X, batch_size = 128, verbose = 0)
        y_pred_labels = np.argmax(y_pred, axis = 1)
        y_pred_labels = y_pred_labels.tolist()
        print("Predicted clips:", y_pred_labels)

        # Find the majority number (most common label)
        if y_pred_labels:  # Ensure y_pred_labels is not empty
            majority_label = Counter(y_pred_labels).most_common(1)[0][0]
            label_list.append(majority_label)

    print("done predict clips")

    # Convert both to numpy arrays
    predicted_labels = np.array(label_list)
    print("Predicted Labels:", predicted_labels)
    true_labels = np.argmax(testY, axis = 1)
    print("True Labels:", true_labels)

    # results are a list of labels.
    # print("Predicted Labels:", y_pred_labels.tolist())
    # print("True Labels:", y_true_labels.tolist())

    # exit(0)

    # Calculate accuracy
    score = np.mean(predicted_labels == true_labels)
    print("Accuracy:", score)

    # Print the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print(conf_matrix / conf_matrix.astype("float").sum(axis = 1))

    score = score * 100.0
    scores.append(score)

    # Accumulate confusion matrices
    if overall_conf_matrix is None:
        overall_conf_matrix = conf_matrix
    else:
        overall_conf_matrix += conf_matrix

# summarize results
ave_acc = processor.summarize_results(scores)
print(f"Overall Confusion Matrix:\n{overall_conf_matrix}")

# Plot heatmap for the overall confusion matrix
plt.figure(figsize = (10, 7))
sns.heatmap(overall_conf_matrix, annot = True, fmt = "d", cmap = "Blues", cbar = True)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title(f"Overall Confusion Matrix Heatmap\n{ave_acc}")
plt.show()

overall_conf_matrix = overall_conf_matrix / overall_conf_matrix.sum(axis = 1,
                                                                    keepdims = True) * 100
# Plot heatmap for the overall confusion matrix
plt.figure(figsize = (10, 7))
sns.heatmap(overall_conf_matrix, annot = True, fmt = ".2f", cmap = "Blues", cbar = True)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title(f"Overall Confusion Matrix Heatmap percentage\n{ave_acc}")
plt.show()

    # self.to_csv(ave_acc, overall_conf_matrix)


# Usage example:
#processor = GestureDataProcessor()

#processor.run_experiment()
