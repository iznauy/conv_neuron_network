from Loss import *
from Layers import *

class Neuron_network(object):

    def __init__(self, hidden_layers_config, output_dim, output_config=None,loss=Softmax_loss, seed=None):

        self.seed = seed
        self.layers_count = len(hidden_layers_config)
        self.layers = []
        self.output_dim = output_dim

        if issubclass(loss, Loss):
            self.loss = loss()
        else:
            raise TypeError("loss Class should be subclass of Loss")

        for layer_config in hidden_layers_config:
            dim_in = layer_config['input_dim']
            dim_out = layer_config['output_dim']
            layer_class = layer_config['layer']
            another_config = layer_config.get('config', {})
            layer = layer_class(dim_in, dim_out, **another_config)

            if len(self.layers) > 0:
                if dim_in != self.layers[-1].dim_out:
                    raise ValueError("The input dimension of each layer must equals to the output"
                                     " dimension of previous layer!")
            self.layers.append(layer)

        output_layer = Affine_layer(self.layers[-1].dim_out, self.output_dim, **(output_config if output_config is not None else {}))
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

