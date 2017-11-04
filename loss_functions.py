import numpy as np


class Loss(object):

    def loss(self, x, y):
        raise TypeError("method 'loss' has not implemented!")


class Svm_loss(Loss):

    def loss(self, x, y):
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

class Softmax_loss(Loss):

    def loss(self, x, y):
        normal_shift = x - np.max(x, axis=1, keepdims=True)
        log_probs = normal_shift - np.log(np.sum(normal_shift, axis=1, keepdims=True))
        probs = np.exp(log_probs)
        N = x.shape[0]
        loss = -np.sum(log_probs[np.arange(N), y]) / N
        dx = probs
        dx[np.arange(N), y] -= 1
        dx /= N
        return loss, dx

