import numpy as np


class Loss(object):

    def loss(self, x, y):
        raise TypeError("method 'loss' has not implemented!")


class Svm_loss(Loss):

    def loss(self, x, y):
        N = x.shape[0]
        correct_class_scores = x[np.arange(N), y]
        margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
        margins[np.arange(N), y] = 0
        loss = np.sum(margins) / N
        num_pos = np.sum(margins > 0, axis=1)
        dx = np.zeros_like(x)
        dx[margins > 0] = 1
        dx[np.arange(N), y] -= num_pos
        dx /= N
        return loss, dx


class Softmax_loss(Loss):

    def loss(self, x, y):
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        N = x.shape[0]
        loss = -np.sum(log_probs[np.arange(N), y]) / N
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        return loss, dx


class L2_loss(Loss):

    def loss(self, x, y):
        loss = np.sum((y - x) ** 2)
        dx = np.sum(2 * (x - y))
        return loss, dx