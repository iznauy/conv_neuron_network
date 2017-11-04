from loss_functions import *
from layers import *

class Neuron_network(object):

    def __init__(self, hidden_layers_config, output_dim, loss=Softmax_loss, learning_rate=1e-2,
                 weight_scale=1e-2, seed=None):

        self.weight_scale = weight_scale
        self.seed = seed
        self.layers_count = len(hidden_layers_config)
        self.layers = []
        self.output_dim = output_dim
        self.learning_rate = 1e-2

        if issubclass(loss, Loss):
            self.loss = loss()
        else:
            raise TypeError("loss Class should be subclass of Loss")

        for layer_config in hidden_layers_config:
            dim_in = layer_config['input_dim']
            dim_out = layer_config['output_dim']
            layer_class = layer_config['layer']
            layer = layer_class(dim_in, dim_out, self.weight_scale, self.learning_rate)
            if len(self.layers) > 0:
                if dim_in != self.layers[-1].dim_out:
                    raise ValueError("The input dimension of each layer must equals to the output dimension of previous layer!")
            self.layers.append(layer)

        output_layer = Affine_layer(self.layers[-1].dim_out, output_dim, weight_scale, learning_rate)
        self.layers.append(output_layer)

        if seed:
            np.random.seed(seed)


if __name__ == '__main__':
    hlc = []
    hlc1 = {'input_dim': 1000, 'output_dim': 100, 'layer': Affine_layer}
    hlc2 = {'input_dim': 120, 'output_dim': 120, 'layer': Relu_layer}
    hlc.append(hlc1)
    hlc.append(hlc2)
    Neuron_network(hlc, 10)