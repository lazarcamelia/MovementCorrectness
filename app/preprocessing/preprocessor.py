import numpy as np

from preprocessing.helpers import uiprmd_helper, intellirehab_helper, vizualizer_helper
from preprocessing.helpers.preprocessing_helper import z_score_normalization, make_equal_length

class Preprocessor:
    def __init__(self, dataset_name, dataset_path) -> None:
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.x_data = []
        self.labels = []
        self.movement_ids = []
        self.subject_ids = []
        self.exercise_ids = []
        self.target_length = 120
        self.x_train = []
        self.x_test = []


    def preprocess(self):
        self.load()
        data_3d = self.process_data_to_3d()
        self.x_data = self.normalize_sequence_length(data_3d)
        print("Input data shape: ", len(self.x_data), len(self.x_data[0]), len(self.x_data[0][0]), len(self.x_data[0][0][0]))
        self.x_data = z_score_normalization(self.x_data)
        print("Input data shape: ", len(self.x_data), len(self.x_data[0]), len(self.x_data[0][0]), len(self.x_data[0][0][0]))

        if self.dataset_name == "intellirehab":
            self.x_data, self.labels = intellirehab_helper.downsampling(self.x_data, self.labels)

            print("After downsampling: ", len(self.x_data), len(self.x_data[0]), len(self.x_data[0][0]),
                  len(self.x_data[0][0][0]))

        # self.vizualize_3d_data_from_list(self.x_data)

    def process_data_to_3d(self):
        exercises_3d = []
        for exercise in self.x_data:
            exercise_3d = []
            for frame in exercise:
                output_frame = self.convert_frame_to_2d(frame)
                exercise_3d.append(output_frame)
            exercises_3d.append(exercise_3d)

        return exercises_3d

    def normalize_sequence_length(self, data_3d):
        new_data = []
        for timeseries in data_3d:
            new_exercise = make_equal_length(timeseries, self.target_length)
            new_data.append(new_exercise)
        return new_data

    def load(self):
        if self.dataset_name == "uiprmd":
           self.load_uiprmd()
        else:
            self.load_intellirehab()

    def load_intellirehab(self):
        print("Processing intellirehab")
        x_data, labels = intellirehab_helper.read_data(self.dataset_path)
        self.x_data = x_data
        self.labels = labels
        # self.movement_ids = movement_ids
        # self.subject_ids = subject_ids
        # self.exercise_ids = exercise_ids

        print("Finished processing")

        # self.vizualize_data()

    def load_uiprmd(self):
        print("Processing UI-PRMD")
        x_data, labels, movement_ids, subject_ids, exercise_ids = uiprmd_helper.read_data(self.dataset_path)

        self.x_data = x_data
        self.labels = labels
        self.movement_ids = movement_ids
        self.subject_ids = subject_ids
        self.exercise_ids = exercise_ids

        print("Finished processing UI-PRMD")

        # self.vizualize_data()

    # def vizualize_data(self):
    #     output_data = []
    #     for frame in self.x_data[0]:
    #         output_frame = self.convert_frame_to_2d(frame)
    #         output_data.append(output_frame)
    #
    #
    #     vizualizer_helper.plot_frame_wrapper_single(output_data)

    def vizualize_data_from_list(self, data):
            output_data = []
            for frame in data[0]:
                output_frame = self.convert_frame_to_2d(frame)
                output_data.append(output_frame)

            vizualizer_helper.plot_frame_wrapper_single(output_data)

    def vizualize_3d_data_from_list(self, data):
        vizualizer_helper.plot_frame_wrapper_single(data[2])

    def convert_frame_to_2d(self, input_list):
        output_list = []
        for i in range(0, len(input_list), 3):
            sublist = input_list[i:i + 3]
            output_list.append(sublist)
        return output_list
