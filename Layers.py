import numpy as np


class Layer(object):

    def __init__(self, dim_in, dim_out, **config):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.cache = {}
        self.grad = {}

    def forward(self, x):
        raise TypeError("method 'forward' has not implemented!")

    def backward(self, dout):
        raise TypeError("method 'forward' has not implemented!")

    def _update(self):
        raise TypeError("method '_update' has not implemented!")


class Affine_layer(Layer):

    def __init__(self, dim_in, dim_out, **config):
        Layer.__init__(self, dim_in, dim_out)
        self.weight_scale = config.get('weight_scale', 1e-2)
        self.w = np.random.randn(dim_in, dim_out) * self.weight_scale
        self.b = np.zeros(dim_out)
        self.learning_rate = config.get('learning_rate', 1e-2)
        self.reg = config.get('reg', 1e-4)

    def forward(self, x):
        out = x.dot(self.w) + self.b
        self.cache['x'] = x
        return out

    def backward(self, dout):
        dx = dout.dot(self.w.T)
        self.grad['w'] = self.cache['x'].T.dot(dout) + self.w * self.reg
        self.grad['b'] = np.sum(dout, axis=0)
        self._update()
        return dx

    def _update(self):
        self.w -= self.learning_rate * self.grad['w']
        self.b -= self.learning_rate * self.grad['b']


class Relu_layer(Layer):
    def __init__(self, dim_in, dim_out, **config):
        Layer.__init__(self, dim_in, dim_out)
        if dim_in != dim_out:
            raise ValueError("In relu layer, the input dimension must equals the output dimension!")

    def forward(self, x):
        self.cache['x'] = x
        out = np.maximum(0, x)
        return out

    def backward(self, dout):
        dx = np.copy(dout)
        dx[self.cache['x'] <= 0] = 0
        return dx

    def _update(self):
        pass

