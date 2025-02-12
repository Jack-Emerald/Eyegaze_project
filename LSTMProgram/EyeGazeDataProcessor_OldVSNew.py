import csv
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from keras import Sequential
from keras import layers
from keras import utils
from keras import Model

from sklearn.metrics import confusion_matrix


class GestureDataProcessor:
    def __init__(self):
        self.feature_match = {"LeftRight": 1, "UpDown": 2, "triangle": 3, "rectangle": 4}
        self.gesture_name = ["LeftRight", "rectangle", "UpDown", "triangle"]
        self.loaded_x = list()
        self.loaded_y = list()
        self.testFile = list()
        self.trainFile = list()
        self.stepLen = '32'
        self.totalAcc = float()

        self.cross_validation()




    def cross_validation(self):
        # Get all possible combinations of 8 numbers from the list of 1 to 10
        self.trainFile = [[1,2,3,4,5,6,7,8,9,10]]
        self.testFile = [[11]]


    def generate_graph(self, file_path):
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

        step = 4*int(self.stepLen)

        # Split the combined_list into sublists of size 448 (7 * 64)
        for i in range(0, len(combined_list), step):
            time_step_data = combined_list[i:i + step]
            if len(time_step_data) == step:
                sublist = [time_step_data[i:i + 4] for i in range(0, len(time_step_data), 4)]
                sublists.append(sublist)

        return sublists

    def combine_same_gesture(self, gesture_name, group):
        with open(f'x_{group}.csv', mode='a', newline='') as file, open(f'y_{group}.csv', mode='a', newline='') as la_file:
            writer = csv.writer(file)
            la_writer = csv.writer(la_file)
            for i in range(1, 9):
                file_path = f"{gesture_name}{i}.txt"
                data_list = self.generate_graph(file_path)

                for sub_list in data_list:
                    new_block = " " + ' '.join(map(str, sub_list))
                    writer.writerow([new_block])
                    la_writer.writerow([self.feature_match[gesture_name]])

                print(f"Processed file {i} for gesture {gesture_name}")

    def load_group(self, gesture_names, group, combination_index):
        if group == "train":
            range_domain = self.trainFile[combination_index]
        elif group == "test":
            range_domain = self.testFile[combination_index]

        for gesture_name in gesture_names:

            for i in range_domain:
                file_path = f"{gesture_name}{i}.txt"
                data_list = self.generate_graph(file_path)

                for step in data_list:
                    self.loaded_y.append([self.feature_match[gesture_name]])

                self.loaded_x += data_list

        loaded_data_x = np.array(self.loaded_x)
        loaded_data_y = np.array(self.loaded_y)
        #print(loaded_data_x)
        #print(loaded_data_y)
        self.loaded_y = list()
        self.loaded_x = list()

        return loaded_data_x, loaded_data_y

    def load_dataSet(self, combination_index):
        trainX, trainY = self.load_group(self.gesture_name, "train", combination_index)
        print(trainX.shape, trainY.shape)
        testX, testY = self.load_group(self.gesture_name, "test", combination_index)
        print(testX.shape, testY.shape)

        # zero-offset class values
        trainY = trainY - 1
        testY = testY - 1
        # one hot encode y
        trainY = utils.to_categorical(trainY)
        testY = utils.to_categorical(testY)
        print(trainX.shape, trainY.shape, testX.shape, testY.shape)
        return trainX, trainY, testX, testY

    def build_attention_model(self, n_timesteps, n_features, n_outputs):
        # Input layer
        inputs = layers.Input(shape = (n_timesteps, n_features))

        # LSTM layer with return_sequences to apply attention
        lstm_out = layers.LSTM(100, return_sequences = True)(inputs)

        # Attention mechanism
        attention = layers.Attention()([lstm_out, lstm_out])

        # Apply dense layer to attention output
        attention_out = layers.Dense(100, activation = 'relu')(attention)

        # Dropout layer for regularization
        attention_out = layers.Dropout(0.5)(attention_out)

        # Final output layer with softmax
        output = layers.Dense(n_outputs, activation = 'softmax')(attention_out)

        # Build and compile model
        model = Model(inputs = inputs, outputs = output)
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        return model

    def evaluate_model(self, trainX, trainy, testX, testy):

        verbose, epochs, batch_size = 0, 15, 64
        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
        model = Sequential()
        model.add(layers.LSTM(100, input_shape = (n_timesteps, n_features)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(100, activation = 'relu'))
        model.add(layers.Dense(n_outputs, activation = 'softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        # fit network
        model.fit(trainX, trainy, epochs = epochs, batch_size = batch_size, verbose = verbose)
        # evaluate model
        _, accuracy = model.evaluate(testX, testy, batch_size = batch_size, verbose = 0)

        # Predict the test data and calculate the confusion matrix
        y_pred = model.predict(testX, batch_size = batch_size, verbose = 0)
        y_pred_labels = np.argmax(y_pred, axis = 1)
        y_true_labels = np.argmax(testy, axis = 1)

        # Print the confusion matrix
        conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
        #print(f"Confusion Matrix:\n{conf_matrix}")

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

    # run an experiment
    def run_experiment(self, repeats = 10):
        for index in range(len(self.trainFile)):
            print(index)
            trainX, trainy, testX, testy = self.load_dataSet(index)
            # repeat experiment
            scores = list()
            overall_conf_matrix = None

            for r in range(repeats):
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

            self.to_csv(ave_acc, overall_conf_matrix)

        print(f"Average acc is {self.totalAcc/len(self.trainFile)}%")



# Usage example:
processor = GestureDataProcessor()

processor.run_experiment()
