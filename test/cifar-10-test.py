import cPickle

from network.Neuron_network import *
from preprocessing.Preprocessing import *


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

path = '/Users/iznauy/Downloads/cifar-10-batches-py/data_batch_1'

if __name__ == '__main__':

    data = unpickle(path)

    train_in = normalization(data['data'][:9000])
    train_labels = np.array(data['labels'][:9000])

    val_in = normalization(data['data'][9000:])
    val_labels = np.array(data['labels'][9000:])
    layers = []
    layers1 = {'input_dim': 3072, 'output_dim': 1000, 'layer': Affine_layer}
    layers_bn1 = {'input_dim': 1000, 'output_dim': 1000, 'layer': BatchNorm_layer}
    layers2 = {'input_dim': 1000, 'output_dim': 1000, 'layer': ReLU_layer}
    layers_dp1 = {'input_dim': 1000, 'output_dim': 1000, 'layer': Dropout_layer}
    layers3 = {'input_dim': 1000, 'output_dim': 100, 'layer': Affine_layer}
    layers_bn2 = {'input_dim': 100, 'output_dim': 100, 'layer': BatchNorm_layer}
    layers4 = {'input_dim': 100, 'output_dim': 100, 'layer': ReLU_layer}
    layers_dp2 = {'input_dim': 100, 'output_dim': 100, 'layer': Dropout_layer}
    layers.append(layers1)
    layers.append(layers_bn1)
    layers.append(layers2)
    layers.append(layers_dp1)
    layers.append(layers3)
    layers.append(layers_bn2)
    layers.append(layers4)
    layers.append(layers_dp2)

    model = Neuron_network(layers, 10)
    for i in range(3000):
        mini_batch_index = np.random.randint(0, 9000, 100)
        mini_batch_in = train_in[mini_batch_index]
        mini_batch_labels = train_labels[mini_batch_index]
        loss = model.train(mini_batch_in, mini_batch_labels)
        if i % 100 == 0 or i == 2999:
            print i, " ", loss

    print 'In predict ', np.sum((model.predict(val_in) == val_labels)) * 1.0 / val_labels.shape[0]
    print 'In train ', np.sum((model.predict(train_in) == train_labels)) * 1.0 / train_labels.shape[0]