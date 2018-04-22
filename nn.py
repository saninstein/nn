import numpy as np
import matplotlib.pyplot as plt


def func(t):
    return np.cos(np.exp(t))

def noise(f):
    return f + np.random.uniform(f * -0.2, f * 0.2)

class NN:
    epoch_len = 100
    E = 0.07
    A = 0.3
    def __init__(self, layers):
        self.neurons = []
        self.layers = layers
        prev = None
        for count_n in layers:
            if prev:
                # self.neurons[0][0] = [[w], [input], [output], [delta W]]
                self.neurons.append([[np.random.uniform(-0.5, 0.5, prev), None, None, np.zeros(prev)] for x in range(count_n)])
            prev = count_n

    def train(self, dataset):
        for i in range(NN.epoch_len):
            outs = []
            for x, y in zip(dataset[0], dataset[1]):
                out = self.forward_propagation(x)[0]
                outs.append(out)
                error = (y - out) ** 2
                # print(i, y, out, error)

                delta = np.array([error * NN.f_deriv(out)])
                for idx, layer in enumerate(reversed(self.neurons)):
                    rev_neurons = list(reversed(self.neurons))
                    if idx == len(self.neurons) - 1:
                        grad = delta * x
                    else:
                        grad = delta * [n[2] for n in rev_neurons[idx + 1]]
                        delta = np.array([NN.f_deriv(n[2]) * np.sum(n[0] * delta) for n in rev_neurons[idx + 1]])

                    delta_w = NN.E * grad + NN.A * np.array([n[3] for n in layer])
                    for dw, n in zip(delta_w, layer):
                        n[0] += dw
                        n[3] = dw
            print(i, np.sqrt(np.sum((dataset[1] - outs) ** 2)/len(outs)))
            break


    def forward_propagation(self, outputs):
        for layer in self.neurons:
            inputs = np.array([np.sum(neuron[0] * outputs) for neuron in layer])
            outputs = np.array([NN.f_actiavation(input) for input in inputs])

            for neuron, input, output in zip(layer, inputs, outputs):
                neuron[1] = input
                neuron[2] = output
        return outputs

    @classmethod
    def f_actiavation(cls, x):
        # return np.tanh(x)
        return 1 / (1 + np.exp(-x))

    @classmethod
    def f_deriv(cls, out):
        # return 1 - out ** 2
        return (1 - out) * out

    def fit():
        pass


step = 0.01


x_list = np.arange(2, 4 + step, step)

x = []
y = []

for i in range(len(x_list) - 5):
    x.append(x_list[i:i + 5])
    y.append(func(x_list[i + 4]))

x, x_test = np.split(x, [int(0.75 * len(x))])
y, y_test = np.split(y, [int(0.75 * len(y))])
dataset = (np.array(x), np.array([noise(f) for f in y]))


nn = NN([5, 5, 1])
nn.train(dataset)

test = [(x[-1], nn.forward_propagation(x)[0]) for x, y in zip(x_test, y_test)]

plt.plot([x[0] for x in test], [x[1] for x in test], color='g')
plt.plot([x[-1] for x in x_test], y_test, color='b')
plt.show()