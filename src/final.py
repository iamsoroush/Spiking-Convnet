from brian2 import *
import mnist
from src.plottools import plot_weights
from time import time
from src.testingtools import test_network


################################################################
# generate train data for 0,2,3,8
################################################################
train_data, train_labels = mnist.train_images(), mnist.train_labels()
print('train data got!')


################################################################
# simulation coefficients
################################################################
N_input = 28*28
N_exc = 50
single_train = 350*ms
resting_time = 150*ms
runtime = single_train + resting_time


################################################################
# neuron model coefficients
################################################################
tauNMDA = 100*ms
tauNMDArise = 10*ms
tauAMPA = 5*ms
vreset = -65*mV
vrest = -65*mV
tau = 50*ms
vth = -52 * mV
tauth = 10*ms
t_refractory = 2*ms
neuron_model = '''
dv/dt = -(v - vreset)/tau + I1/tau : volt (unless refractory)
dI1/dt = -I1/tauAMPA : volt
dvthadaptive/dt = -(vthadaptive - vth)/tauth : volt
'''


################################################################
# synapse model coes
################################################################
tautrace = 15*ms
mu = 0.9
offset = 0.4
eta = 0.01
wmax = 1
inhibitory_weight = 1
synapse_model = '''
w : 1
dTrace/dt = -Trace/tautrace : 1 (clock-driven)
'''
synapse_onpre = '''
I1_post += 20*w*mV
Trace += 1
'''
synapse_onpost = '''
w += eta*(Trace - offset)*((wmax - w)**mu)
vthadaptive += 1*mV
'''


################################################################
# initialization
################################################################
input_layer = PoissonGroup(N_input, rates=[0 for i in range(N_input)] * Hz)
excitatory_layer = NeuronGroup(N_exc, neuron_model, threshold='v>vthadaptive',
                               reset='v=-65*mV', refractory=t_refractory, method='exact')
S_1 = Synapses(input_layer, excitatory_layer, model=synapse_model, on_pre=synapse_onpre, on_post=synapse_onpost)
S_inhibitory = Synapses(excitatory_layer, excitatory_layer, on_pre='I1_post -= 10*mV')
S_1.connect()
S_1.w = 'rand()/10'
S_inhibitory.connect(condition='i != j')
monitor_exc = SpikeMonitor(excitatory_layer)


################################################################
# running
################################################################
t_time = time()
for ind, sample in enumerate(train_data):
    vth_last = excitatory_layer.vthadaptive
    input_layer.rates = [0 for _ in range(N_input)] * Hz
    run(resting_time)
    if ind % 50 == 0:
        print('starting input {}'.format(ind))
    excitatory_layer.vthadaptive = vth_last
    input_layer.rates = sample * Hz
    input_max = 63.75
    while True:
        run(single_train)
        if monitor_exc.num_spikes > 10:
            break
        else:
            new_sample = [item * (input_max + 32) / input_max for item in sample]
            sample = new_sample
            input_max += 32
            input_layer.rates = sample * Hz
print('training time :{}'.format(time() - t_time))


################################################################
# plotting weights
################################################################
weights = []
for i in range(25):
    w_list = S_1.w[:, i].tolist()
    w = []
    for j in w_list:
        if j >= 0:
            w.append(j)
        else:
            w.append(0)
    weights.append(w)
plot_weights(weights, 'weights1_25')
weights = []
for i in range(25, 50):
    w_list = S_1.w[:, i].tolist()
    w = []
    for j in w_list:
        if j >= 0:
            w.append(j)
        else:
            w.append(0)
    weights.append(w)
plot_weights(weights, 'weights26_50')


################################################################
# testing
################################################################
test_network(excitatory_layer.vthadaptive, S_1.w, train_data, train_labels)
