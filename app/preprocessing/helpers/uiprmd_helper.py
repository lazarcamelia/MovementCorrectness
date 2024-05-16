import pandas as pd
import numpy as np
import os

def read_data(dataset_path):

    all_files = os.listdir(dataset_path)

    x_data = []
    labels = []
    movement_ids = []
    subject_ids = []
    exercise_ids = []

    for filename in all_files:
        if 'txt' not in filename:
            continue

        movement_id, subject_id, exercise_id, label = extract_info_from_filename(filename)
        current_exercise = read_txt_file(dataset_path + filename)

        x_data.append(current_exercise)
        movement_ids.append(movement_id)
        subject_ids.append(subject_id)
        exercise_ids.append(exercise_id)
        labels.append(label)

    labels_numpy = np.array(labels)

    return x_data, labels_numpy, movement_ids, subject_ids, exercise_ids


def extract_info_from_filename(filename):
    '''
        split the filename and return metadata:
            - movement_id
            - subject_id
            - exercise_id
    '''
    pieces = filename.split("_")

    movement_id = pieces[0]
    subject_id = pieces[1]
    exercise_id = pieces[2]

    # the label 1 is for correct, 2 is for incorrect
    label = 0

    # if _inc is added at the end, the movement is incorrect
    if len(pieces) == 5:
        label = 1

    return movement_id, subject_id, exercise_id, label


def read_txt_file(filename):
    df = pd.read_csv(filename, sep = ',', header=None, index_col=False)
    return df.values
