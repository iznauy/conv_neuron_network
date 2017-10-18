import numpy as np

def svm_loss(x, y):
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, None] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    num_count = np.sum(dx, axis=1)
    dx[np.arange(N), y] -= num_count
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    pass