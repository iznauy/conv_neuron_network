import numpy as np


def affine_forward(x, w, b):
    N = x.shape[0]
    D = x.size / N
    x_vec = x.reshape((N, D))
    out = x_vec.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache
    N = x.shape[0]
    D = x.size / N
    x_vec = x.reshape((N, D))
    dx = np.dot(dout, w.T)
    dw = np.dot(x_vec.T, dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = dout.copy()
    dx[x <= 0] = 0
    return dx


def weak_relu_forward(x, r):
    out = np.maximum(0, x)
    cache = (x, r)
    out[x <= 0] = r * x[x <= 0]
    return out, cache


def weak_relu_backward(dout, cache):
    x, r = cache
    dx = dout.copy()
    dx[dx <= 0] = r * dx[dx <= 0]
    return dx


def affine_relu_forward(x, w, b):
    temp_out, affine_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(temp_out)
    cache = (affine_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    affine_cache, relu_cache = cache
    dtemp = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(dtemp, affine_cache)
    return dx, dw, db

