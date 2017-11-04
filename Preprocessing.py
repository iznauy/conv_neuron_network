import numpy as np

def normalization(x, dtype=np.float32):
    """
    Normalize the input data
    :param x: 2-d array (N, D). N represents the number of training examples,
                D means the amount of features of each example.
    :return: x with zero-means
    """
    x = x.astype(dtype)
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    return (x - means) / stds
