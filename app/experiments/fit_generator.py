import numpy as np

def custom_fit_generator(data, labels):
    while True:
        for sequence, label in zip(data, labels):
            # Reshape sequence to match the input shape of the model
            x = np.expand_dims(sequence, axis=0)
            y = np.array([[label]])  # Convert label to array and add batch dimension
            yield x, y
