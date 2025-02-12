import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import csv
from pandas import read_csv
import numpy as np

def generateGraph(file_path):
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


                combined_list = combined_list + [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, ori_w]

    # Split the combined_list into sublists of size 448, which is 7*64. 64 is time step size.
    # [[7 for 64 pairs, each pair is a list],[...]....]
    for i in range(0, len(combined_list), 448):
        TimeStepData = combined_list[i:i + 448]
        if len(TimeStepData) == 448:  # Only append if the sublist length is exactly 448
            sublist = [TimeStepData[i:i + 7] for i in range(0, len(TimeStepData), 7)] # each 7 value pairs as a sublist.
                                                                                      # it represents the eye gaze data change in one step.
                                                                                      # And each step should correspond to a label.
            sublists.append(sublist) # gather sublists in to one list

    return sublists

def combine_same_gesture(Gesturename, group):
    Feature_match = {"LeftRight":'1', "rectangle":'4',"UpDown":'2',"triangle":'3'}
    with open('x_'+group+'.csv', mode = 'a', newline = '') as file, open('y_'+group+'.csv', mode='a', newline='') as la_file:
        writer = csv.writer(file)
        la_writer = csv.writer(la_file)
        for i in range(1, 9):
            file_path = Gesturename + str(i) + ".txt"
            data_list = generateGraph(file_path) # a list of steps

            for sub_list in data_list:
                new_block = " "+' '.join(map(str, sub_list))
                writer.writerow([new_block])
                la_writer.writerow(Feature_match[Gesturename])

            print(i)

def load_group(Gesturename):
    Feature_match = {"LeftRight":1, "rectangle":4,"UpDown":2,"triangle":3}
    loaded_x = list()
    loaded_y = list()

    for i in range(1, 9):
        file_path = Gesturename + str(i) + ".txt"
        data_list = generateGraph(file_path) # a list of steps
        for step in data_list:
            loaded_y.append([Feature_match[Gesturename]])

        loaded_x += data_list

    loaded_data_x = np.array(loaded_x)
    loaded_data_y = np.array(loaded_y)
    print(loaded_data_y)

    return loaded_data_x, loaded_data_y


def read_all_features():
    featre_list = ["LeftRight", "UpDown", "triangle", "rectangle"]
    for feature in featre_list:
        combine_same_gesture(feature, 'train')



#read_all_features()
load_group("LeftRight")