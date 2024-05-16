from preprocessing.preprocessor import Preprocessor
from experiments.LSTM_experiments import LSTMExperiments
from experiments.CNN_experiments import CNNExperiments
if __name__ == '__main__':
    datasets = [
        ["kimore", "/Users/camelialazar/Desktop/Master/Disertatie/ProiectNou/MovementCorrectness/app/data/kimore/"],
        ["intellirehab", "/Users/camelialazar/Desktop/Master/Disertatie/ProiectNou/MovementCorrectness/app/data/intellirehab/"],
        ["uiprmd", "/Users/camelialazar/Desktop/Master/Disertatie/ProiectNou/MovementCorrectness/app/data/ui-prmd/"]
    ]

    KIMORE = 0
    INTELLIREHAB = 1
    UIPRMD = 2

    dataset_name = datasets[INTELLIREHAB][0]
    dataset_path = datasets[INTELLIREHAB][1]

    # running forf UI-PRMD
    preprocess = Preprocessor(dataset_name, dataset_path)
    preprocess.preprocess()

    # lstm_experiment = LSTMExperiments(preprocess.x_data, preprocess.labels)
    # lstm_experiment.run_experiment()

#     cnn experiments
    cnn_experiment = CNNExperiments(preprocess.x_data, preprocess.labels)
    # cnn_experiment.create_resnet_model()
    # cnn_experiment.create_resnet_model_early_stopping()
    cnn_experiment.create_resnet_model_with_weighthed_loss()
    # cnn_experiment.create_inception_model()
