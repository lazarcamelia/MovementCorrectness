import numpy as np

from preprocessing.helpers import uiprmd_helper, intellirehab_helper, vizualizer_helper

class Preprocessor:
    def __init__(self, dataset_name, dataset_path) -> None:
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.x_data = []
        self.labels = []
        self.movement_ids = []
        self.subject_ids = []
        self.exercise_ids = []

    def get_data(self):
       return self.data

    def preprocess(self):
        if self.dataset_name == "kimore":
            self.preprocess_kimore()
        elif self.dataset_name == "uiprmd":
           self.preprocess_uiprmd()
        else:
            self.preprocess_intellirehab()

    def preprocess_kimore(self):
      pass

    def preprocess_intellirehab(self):
        print("Processing intellirehab")
        x_data, labels_numpy, subject_ids, exercise_ids, date_ids, repetitions, positions, labels = intellirehab_helper.read_data(self.dataset_path)
        self.x_data = x_data
        self.labels = labels
        # self.movement_ids = movement_ids
        self.subject_ids = subject_ids
        self.exercise_ids = exercise_ids

        # self.vizualize_data()

    def preprocess_uiprmd(self):
        print("Processing UI-PRMD")
        x_data, labels, movement_ids, subject_ids, exercise_ids = uiprmd_helper.read_data(self.dataset_path)

        self.x_data = x_data
        self.labels = int(labels)
        self.movement_ids = movement_ids
        self.subject_ids = subject_ids
        self.exercise_ids = exercise_ids

        # self.vizualize_data()


    def vizualize_data(self):
        print(len(self.x_data))
        print(len(self.x_data[0]))
        print(len(self.x_data[0][0]))

        output_data = []
        for frame in self.x_data[0]:
            output_frame = self.convert_frame_to_2d(frame)
            output_data.append(output_frame)

        print(len(output_data))
        print(len(output_data[0]))
        print(len(output_data[0][0]))


        vizualizer_helper.plot_frame_wrapper_single(output_data)


    def convert_frame_to_2d(self, input_list):
        output_list = []
        for i in range(0, len(input_list), 3):
            sublist = input_list[i:i + 3]
            output_list.append(sublist)
        return output_list


