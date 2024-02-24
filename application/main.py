from preprocessing import Preprocessor

if __name__ == '__main__':
    datasets = [
        ["kimore", "/Users/camelialazar/Desktop/Master/Disertatie/Proiect Nou/MovementCorrectness/application/data/kimore"],
        ["intellirehab", "/Users/camelialazar/Desktop/Master/Disertatie/Proiect Nou/MovementCorrectness/application/data/intellirehab"],
        ["uiprmd", "/Users/camelialazar/Desktop/Master/Disertatie/Proiect Nou/MovementCorrectness/application/data/ui-prmd"]
    ]
    preprocess = Preprocessor(datasets[2][0], datasets[2][1])

    preprocess.preprocess()