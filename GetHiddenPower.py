# GetHiddenPower.py

import numpy as np
import sys
import time
import cPickle
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano

# Create the training data
x, y = cPickle.load(open('OnlyNPs_codeonly.cPickle', 'rb'))
X_train = np.array(x, dtype='float32')
y_train = np.array(y, dtype='float32')

# Specify model
model = RNNTheano(10, 333, hidden_dim = 30)

# Load previously saved model
load_model_parameters_theano('model_parameters_OnlyNPs.npz', model)

#itearte over the full dataset to get the array of hidden states
def get_hidden_states(input_array):
    emptylist = []
    for i in range(len(input_array)):
        _, s = model.forward_propagation(X_train[i])
        emptylist.append(s)
    return np.array(emptylist)
hidden_states = get_hidden_states(X_train)
print hidden_states.shape
    
# Count neurons firing at 1hz and record their idexes
def count_1hz(hidden_states_array, threshold):
    counter = 0
    index_list = []
    a,b,c = hidden_states_array.shape
    for j in range(c):
        if np.all(hidden_states_array[:,1:,j] > threshold):
            counter += 1
            index_list.append(j)
    return counter, index_list
counter_1hz, indexes_1hz = count_1hz(hidden_states, 0.5)
# Count 2hz (at any start point) and record their idexes
def count_2hz_any(hidden_states_array, threshold):
    counter = 0
    index_list = []
    a,b,c = hidden_states_array.shape
    for j in range(c):
        if (np.all(hidden_states_array[:,1:3,j] > threshold) or np.all(hidden_states_array[:,2:4,j] > threshold) or np.all(hidden_states_array[:,3:5,j] > threshold)):
                counter += 1
                index_list.append(j)
    return counter, index_list    
# count 2hz (at any start point) for OnlyNPs12 condition
def count_2hz_any_OnlyNPs12(hidden_states_array, threshold):
    counter = 0
    index_list = []
    a,b,c = hidden_states_array.shape
    for j in range(c):
        if (np.all(hidden_states_array[:,1:3,j] > threshold) or np.all(hidden_states_array[:,2:4,j] > threshold)):
                counter += 1
                index_list.append(j)
    return counter, index_list
# count 2hz (at any start point) for OnlyNPs condition
def count_2hz_any_OnlyNPs(hidden_states_array, threshold):
    counter = 0
    index_list = []
    a,b,c = hidden_states_array.shape
    for j in range(c):
        if (np.all(hidden_states_array[:,1,j] > threshold) or np.all(hidden_states_array[:,2,j] > threshold)):
                counter += 1
                index_list.append(j)
    return counter, index_list
counter_2hz, indexes_2hz = count_2hz_any_OnlyNPs(hidden_states, 0.5)
# Count 3hz (at any start point) and record their idexes
def count_3hz_any(hidden_states_array, threshold):
    counter = 0
    index_list = []
    a,b,c = hidden_states_array.shape
    for j in range(c):
        if (np.all(hidden_states_array[:,1:4,j] > threshold) or np.all(hidden_states_array[:,2:5,j] > threshold)):
                counter += 1
                index_list.append(j)
    return counter, index_list
#counter_3hz, indexes_3hz = count_3hz_any(hidden_states, 0.7)
# Count 3hz for OnlyNPs12 condition (this is equivalent to the Count 4hz for the 4-words conditions)
def count_3hz_any_OnlyNPs12(hidden_states_array, threshold):
    counter = 0
    index_list = []
    a,b,c = hidden_states_array.shape
    for j in range(c):
        if (np.all(hidden_states_array[:,1,j] > threshold) or np.all(hidden_states_array[:,2,j] > threshold) or np.all(hidden_states_array[:,3,j] > threshold)):
                counter += 1
                index_list.append(j)
    return counter, index_list
#counter_3hz, indexes_3hz = count_3hz_any_OnlyNPs12(hidden_states, 0.5)
# Count 4hz (at any start point) and record their idexes
def count_4hz_any(hidden_states_array, threshold):
    counter = 0
    index_list = []
    a,b,c = hidden_states_array.shape
    for j in range(c):
        if (np.all(hidden_states_array[:,1,j] > threshold) or np.all(hidden_states_array[:,2,j] > threshold) or np.all(hidden_states_array[:,3,j] > threshold) or np.all(hidden_states_array[:,4,j] > threshold)):
                counter += 1
                index_list.append(j)
    return counter, index_list
#counter_4hz, indexes_4hz = count_4hz_any(hidden_states, 0.7)

# Return the mean activation across sentences of a series of neurons indexed by a list
#def mean_activation(list_of_neurons):
    
print counter_1hz, indexes_1hz
print counter_2hz, indexes_2hz

x = hidden_states[:,1:,:]
print x.shape
'''
np.savez('hidden_states_OnlyNPs', hidden_states=x)
npzfile = np.load('hidden_states_OnlyNPs.npz')
hs_OnlyNPs = npzfile['hidden_states']
print hs_OnlyNPs.shape
print type(hs_OnlyNPs)
'''





