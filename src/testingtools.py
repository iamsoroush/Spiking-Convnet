from brian2 import *
import mnist


def test_network(vthadaptive, w, train_data, train_labels):
    ################################################################
    # simulation coefficients
    ################################################################
    N_input = 28*28
    N_exc = 50
    single_train = 350*ms
    resting_time = 150*ms

    ################################################################
    # neuron model coefficients
    ################################################################
    t_refractory = 2*ms
    neuron_model = '''
    dv/dt = -(v - vreset)/tau + I1/tau : volt (unless refractory)
    dI1/dt = -I1/tauAMPA : volt
    vthadaptive : volt
    '''

    ################################################################
    # synapse model coefficients
    ################################################################
    synapse_model = '''
    w : 1
    '''
    synapse_onpre = '''
    I1_post += 20*w*mV
    '''

    ################################################################
    # initialization
    ################################################################
    input_layer = PoissonGroup(N_input, rates=[0 for _ in range(N_input)] * Hz)
    excitatory_layer = NeuronGroup(N_exc, neuron_model, threshold='v>vthadaptive',
                                   reset='v=-65*mV', refractory=t_refractory, method='exact')
    S_1 = Synapses(input_layer, excitatory_layer, model=synapse_model, on_pre=synapse_onpre)
    S_1.connect()
    S_1.w = w
    excitatory_layer.vthadaptive = vthadaptive
    monitor_exc = SpikeMonitor(excitatory_layer)

    ################################################################
    # labeling neurons
    ################################################################
    spikes_list = [[0 for _ in range(10)] for _ in range(N_exc)]
    for (ind, sample) in enumerate(train_data):
        label = train_labels[ind]
        input_layer.rates = [0 for _ in range(N_input)] * Hz
        run(resting_time)
        input_layer.rates = sample * Hz
        run(single_train)
        for i in range(N_exc):
            spikes_list[i][label] += monitor_exc.count[i]
    neuron_labels = {}
    for ind, neuron in enumerate(spikes_list):
        neuron_labels[ind] = neuron.index(max(neuron))
    print("neuron lablels : ", neuron_labels)

    ################################################################
    # testing
    ################################################################
    test_data, test_labels = mnist.test_images(), mnist.test_labels()
    wins = 0
    for ind, sample in enumerate(test_data):
        label = test_labels[ind]
        input_layer.rates = [0 for _ in range(N_input)] * Hz
        run(resting_time)
        input_layer.rates = sample * Hz
        run(single_train)
        spikes_counter = [0 for _ in range(10)]
        for i in range(N_exc):
            spikes_counter[neuron_labels[i]] += monitor_exc.count[i]
        if spikes_counter.index(max(spikes_counter)) == label:
            wins += 1
        else:
            continue
    print('wins :{}'.format(wins), 'total :{}'.format(len(test_labels)))
    print('accuracy :{}'.format(wins / len(test_labels)))
