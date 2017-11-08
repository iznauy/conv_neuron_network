import numpy as np


class Layer(object):

    def __init__(self, dim_in, dim_out, **config):
        if type(dim_in) != type(1) or type(dim_out) != type(1):
            raise TypeError("The input dimension or output dimension must be an Integer!")
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.cache = {}
        self.grad = {}


    def forward(self, x, **kwargs):
        raise TypeError("method 'forward' has not implemented!")


    def backward(self, dout, **kwargs):
        raise TypeError("method 'forward' has not implemented!")


    def _update(self):
        raise TypeError("method '_update' has not implemented!")



class BatchNorm_layer(Layer):

    def __init__(self, dim_in, dim_out, **config):

        Layer.__init__(self, dim_in, dim_out)
        if dim_out != dim_in:
            raise ValueError("In Batch normalization layer, the input dimension must equals the output dimension!")

        self.gamma = np.ones(dim_in)
        self.beta = np.zeros(dim_in)
        self.running_mean = np.zeros(dim_in)
        self.running_var = np.zeros(dim_in)
        self.eps = config.get("eps", 1e-5)
        self.momentum = config.get("momentum", 0.9)
        self.learning_rate = config.get('learning_rate', 1e-2)



    def forward(self, x, **kwargs):

        mode = kwargs.get('mode', 'train')
        if mode == 'train':
            sample_mean = np.mean(x, axis=0)
            sample_val = np.mean((x - sample_mean) ** 2, axis=0)
            x_norm = (x - sample_mean) / np.sqrt(sample_val + self.eps)
            out = self.gamma * x_norm + self.beta
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * sample_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * sample_val
            self.cache['x'] = x
            self.cache['sample_mean'] = sample_mean
            self.cache['sample_var'] = sample_val
            self.cache['x_norm'] = x_norm
        elif mode == 'test':
            out = self.gamma * (x - self.running_mean) / np.sqrt(self.running_var + self.eps) + self.beta
        else:
            raise ValueError("InValid forward batchnorm mode!")
        return out


    def backward(self, dout, **kwargs):

        x, sample_mean, sample_var, x_norm = self.cache['x'], self.cache['sample_mean'], self.cache['sample_var'], self.cache['x_norm']
        dx_norm = dout * self.gamma
        d_var = np.sum(-0.5 * (sample_var + self.eps) ** (-1.5) * (x - sample_mean) * dx_norm, axis=0)
        d_mean = np.sum(dx_norm * (-1.0) / np.sqrt(sample_var + self.eps), axis=0) - 2 * np.mean(d_var * (x - sample_mean), axis=0)
        dx = dx_norm / np.sqrt(sample_var + self.eps) + (d_var * 2 * (x - sample_mean) + d_mean) / x.shape[0]
        self.grad['gamma'] = np.sum(dout * x_norm, axis=0)
        self.grad['beta'] = np.sum(dout, axis=0)
        self._update()
        return dx

    def _update(self):
        self.gamma -= self.learning_rate * self.grad['gamma']
        self.beta -= self.learning_rate * self.grad['beta']



class Dropout_layer(Layer):

    def __init__(self, dim_in, dim_out, **kwargs):
        Layer.__init__(self, dim_in, dim_out)
        if dim_in != dim_out:
            raise ValueError("In Dropout layer, the input dimension must equals the output dimension!")
        self.p = kwargs.get("p", 0.5)
        if self.p <= 0 or self.p >= 1:
            raise ValueError("The ratio of dropout must within 0 and 1!")


    def forward(self, x, **kwargs):
        mode = kwargs.get("mode", "train")
        if mode == 'train':
            mask = (np.random.rand(*x.shape) >= self.p) / (1 - self.p)
            out = mask * x
            self.cache['mask'] = mask
        elif mode == 'test':
            out = x
        else:
            raise ValueError("InValid forward dropout mode!")
        return out


    def backward(self, dout, **kwargs):
        dx = None
        mode = kwargs.get("mode", "train")
        if mode == 'train':
            dx = dout * self.cache['mask']
        elif mode == 'test':
            dx = dout
        else:
            raise ValueError("InValid backward dropout mode!")
        return dx


    def _update(self):
        pass



class Affine_layer(Layer):

    def __init__(self, dim_in, dim_out, **config):
        Layer.__init__(self, dim_in, dim_out)
        self.weight_scale = config.get('weight_scale', 1e-2)
        self.w = np.random.randn(dim_in, dim_out) * self.weight_scale
        self.b = np.zeros(dim_out)
        self.learning_rate = config.get('learning_rate', 1e-2)
        self.reg = config.get('reg', 1e-4)


    def forward(self, x, **kwargs):
        out = x.dot(self.w) + self.b
        self.cache['x'] = x
        return out


    def backward(self, dout, **kwargs):
        dx = dout.dot(self.w.T)
        self.grad['w'] = self.cache['x'].T.dot(dout) + self.w * self.reg
        self.grad['b'] = np.sum(dout, axis=0)
        self._update()
        return dx


    def _update(self):
        self.w -= self.learning_rate * self.grad['w']
        self.b -= self.learning_rate * self.grad['b']



class ReLU_layer(Layer):

    def __init__(self, dim_in, dim_out, **config):
        Layer.__init__(self, dim_in, dim_out)
        if dim_in != dim_out:
            raise ValueError("In relu layer, the input dimension must equals the output dimension!")


    def forward(self, x, **kwargs):
        self.cache['x'] = x
        out = np.maximum(0, x)
        return out


    def backward(self, dout, **kwargs):
        dx = np.copy(dout)
        dx[self.cache['x'] <= 0] = 0
        return dx


    def _update(self):
        pass


