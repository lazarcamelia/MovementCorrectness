from preprocessing.preprocessor import Preprocessor
from experiments.LSTM_experiments import LSTMExperiments

if __name__ == '__main__':
    datasets = [
        ["kimore", "/Users/camelialazar/Desktop/Master/Disertatie/ProiectNou/MovementCorrectness/app/data/kimore"],
        ["intellirehab", "/Users/camelialazar/Desktop/Master/Disertatie/ProiectNou/MovementCorrectness/app/data/intellirehab"],
        ["uiprmd", "/Users/camelialazar/Desktop/Master/Disertatie/ProiectNou/MovementCorrectness/app/data/ui-prmd/"]
    ]

    # running forf UI-PRMD
    preprocess = Preprocessor(datasets[2][0], datasets[2][1])
    preprocess.preprocess()

    lstm_experiment = LSTMExperiments(preprocess.x_data, preprocess.labels)

    lstm_experiment.run_experiment()
