import numpy as np


def sgd(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    config.setdefault('velocity', np.zeros_like(dw))
    velocity = config['velocity']
    velocity = config['momentum'] * velocity - config['learning_rate'] * dw
    w += velocity
    config['velocity'] = velocity
    return w, config


def adagrad(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('cache', np.zeros_like(dw))
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('eps', 1e-6)
    w += -config['learning_rate'] * dw / (np.sqrt(config['cache']) + config['eps'])
    return w, config


def rmsprop(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault("cache", np.zeros_like(dw))
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("eps", 1e-6)
    config.setdefault("decay_rate", 0.99)
    config['cache'] = config['cache'] * config['decay_rate'] + (1 - config['decay_rate']) * dw ** 2
    w += -config['learning_rate'] * dw / (np.sqrt(config['cache']) + config['eps'])
    return w, config


def adam(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('eps', 1e-8)
    config.setdefault('m', np.zeros_like(dw))
    config.setdefault('v', np.zeros_like(dw))
    config.setdefault('t', 1)
    beta1, beta2, learning_rate = config['beta1'], config['beta2'], config['learning_rate']
    config['m'] = config['m'] * beta1 + (1 - beta1) * dw
    mt = config['m'] / (1 - beta1 ** t)
    config['v'] = config['v'] * beta2 + (1 - beta2) * (dw ** 2)
    vt = config['v'] / (1 - beta2 ** t)
    w += -learning_rate * mt / (np.sqrt(vt) + config['eps'])
    config['t'] += 1
    return w, config

