from preprocessing.helpers import uiprmd_helper

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
      pass

    def preprocess_uiprmd(self):
        x_data, labels, movement_ids, subject_ids, exercise_ids = uiprmd_helper.read_data(self.dataset_path)

        self.x_data = x_data
        self.labels = labels
        self.movement_ids = movement_ids
        self.subject_ids = subject_ids
        self.exercise_ids = exercise_ids

