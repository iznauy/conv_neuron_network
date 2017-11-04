import cPickle
from Preprocessing import *
from Neuron_network import *

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
    layers2 = {'input_dim': 1000, 'output_dim': 1000, 'layer': Relu_layer}
    layers.append(layers1)
    layers.append(layers2)

    model = Neuron_network(layers, 10, learning_rate=0.5, weight_scale=1e-3)
    for i in range(3000):
        mini_batch_index = np.random.randint(0, 9000, 100)
        mini_batch_in = train_in[mini_batch_index]
        mini_batch_labels = train_labels[mini_batch_index]
        loss = model.train(mini_batch_in, mini_batch_labels)
        if i % 100 == 0 or i == 5999:
            print i, " ", loss

    print 'In predict ', np.sum((model.predict(val_in) == val_labels)) * 1.0 / val_labels.shape[0]
    print 'In predict ', np.sum((model.predict(train_in) == train_labels)) * 1.0 / train_labels.shape[0]