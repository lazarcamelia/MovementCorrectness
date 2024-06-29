from preprocessing.preprocessor import Preprocessor
from experiments.ResNetExperiments import ResNetExperiments
from experiments.InceptionExperiments import InceptionExperiments
from experiments.InceptionTimeExperiments import InceptionTimeExperiments
from experiments.LSTM_experiments import LSTMExperiments
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
    #
    # lstm_experiment = LSTMExperiments(preprocess.x_data, preprocess.labels)
    # lstm_experiment.run_experiment()
    # #
# #     cnn experiments
#     cnn_experiment = ResNetExperiments(preprocess.x_data, preprocess.labels)
#     cnn_experiment.create_resnet_model_with_weighthed_loss()

    # inception_experiment = InceptionExperiments(preprocess.x_data, preprocess.labels)
    # inception_experiment.create_inception_model()

    inceptiontime = InceptionTimeExperiments(preprocess.x_data, preprocess.labels)
    inceptiontime.inception_time_model()
