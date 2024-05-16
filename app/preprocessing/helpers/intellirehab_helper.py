import pandas as pd
import numpy as np
import os


# SubjectID_DateID_GestureLabel_RepetitionNumber_CorrectLabel_Position.txt
# SubjectID: id uniquely identifying the person performing the exercise
# DateID: id identifying the session in which the person was recorded
# GestureLabel: Label identifying the gesture; possible values are from 0 to 8
# RepetitionNumber: Each gesture was repeated several times and this shows the repetition number
# CorrectLabel: A value of 1 represents a gesture labeled as being correctly executed, while a value of 2 is for a gesture labeled as incorrect
# Position: Some of the persons performed the gestures sitting on a chair or wheelchair, while others standing
def read_data(dataset_path):
    all_files = os.listdir(dataset_path)

    x_data = []
    labels = []
    date_ids = []
    subject_ids = []
    exercise_ids = []
    positions = []
    repetitions = []

    index = -1

    for filename in all_files:
        index += 1
        if 'txt' not in filename:
            continue

        subject_id, date_id, exercise_id, repetition, label, position = extract_info_from_filename(filename)
        # the labels = 3 are poorly executed and are not considered as part of the dataset
        if label == 2:
            continue

        current_exercise = read_txt_file(dataset_path + filename)
        # if index == 0:
        #     print(filename)
        #     print(current_exercise)

        x_data.append(current_exercise)
        subject_ids.append(subject_id)
        exercise_ids.append(exercise_id)
        date_ids.append(date_id)
        repetitions.append(repetition)
        positions.append(position)
        labels.append(label)

    labels_numpy = np.array(labels)

    return x_data, labels_numpy, subject_ids, exercise_ids, date_ids, repetitions, positions, labels

def extract_info_from_filename(filename):
    '''
       split the filename and return metadata:
           - subject_id
           - date_id
           - exercise_id
           - repetition_number
           - label
           - position: it can be stand, chair or wheelchair
    '''

    pieces = filename.split("_")

    subject_id = pieces[0]
    date_id = pieces[1]
    exercise_id = pieces[2]
    repetition_number = pieces[3]
    label = int(pieces[4]) - 1

    position_char = pieces[5]

    if position_char == "stand":
        position = 1
    elif position_char == "chair":
        position = 2
    else:
        position = 3

    return subject_id, date_id, exercise_id, repetition_number, label, position


def read_txt_file(filename):
    df = pd.read_csv(filename, sep=',', header=None, index_col=False)
    return df.values
