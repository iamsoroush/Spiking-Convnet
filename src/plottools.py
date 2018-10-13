import matplotlib.pyplot as plt
import matplotlib.pylab as plab


def plot_weights(weights, name):  # 2 ta 5*5
    mymap = plt.get_cmap("Reds")
    fig, axs = plt.subplots(5, 5)
    for (ind, neuron) in enumerate(weights):
        x = [j for j in range(1, 29)]
        for i in range(28):
            y = [28-i]*28
            intensity = neuron[i*28:(i+1)*28]
            axs[ind//5, ind % 5].scatter(x, y, c=intensity, cmap=mymap)

    plab.savefig('{}.jpg'.format(name), bbox_inches='tight')
