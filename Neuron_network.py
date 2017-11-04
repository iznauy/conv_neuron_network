from Loss import *
from Layers import *

class Neuron_network(object):

    def __init__(self, hidden_layers_config, output_dim, loss=Softmax_loss, learning_rate=1e-2,
                 reg=1e-4, weight_scale=1e-2, seed=None):

        self.weight_scale = weight_scale
        self.seed = seed
        self.layers_count = len(hidden_layers_config)
        self.layers = []
        self.output_dim = output_dim
        self.learning_rate = 1e-2
        self.reg = reg

        if issubclass(loss, Loss):
            self.loss = loss()
        else:
            raise TypeError("loss Class should be subclass of Loss")

        for layer_config in hidden_layers_config:
            dim_in = layer_config['input_dim']
            dim_out = layer_config['output_dim']
            layer_class = layer_config['layer']
            layer = layer_class(dim_in, dim_out, self.weight_scale, self.learning_rate, self.reg)

            if len(self.layers) > 0:
                if dim_in != self.layers[-1].dim_out:
                    raise ValueError("The input dimension of each layer must equals to the output dimension of previous layer!")
            self.layers.append(layer)

        output_layer = Affine_layer(self.layers[-1].dim_out, self.output_dim, self.weight_scale,
                                    self.learning_rate, self.reg)
        self.layers.append(output_layer)

        if seed:
            np.random.seed(seed)

    def train(self, x, y):
        return self._loss(x, y)

    def predict(self, x):
        scores = self._loss(x)
        return np.argmax(scores, axis=1)

    def _loss(self, x, y=None):

        mode = 'test' if y is None else 'train'

        for layer in self.layers:
            x = layer.forward(x)
        scores = x

        if mode == 'test':
            return scores

        loss, dout = self.loss.loss(scores, y)
        for i in range(len(self.layers) - 1, -1, -1):
            dout = self.layers[i].backward(dout)

        return loss

