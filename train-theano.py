#! /usr/bin/env python

import numpy as np
import sys
import time
import cPickle
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush() # You want unbuffered output whenever you want to ensure that the output has been written before continuing.
            # ADDED! Saving model oarameters
            #save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        # Optionally evaluate the accuracy
        if (epoch % evaluate_loss_after == 0):
            accuracy = model.calculate_accuracy(X_train, y_train)
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Accuracy after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, accuracy)

        # For each training example (SGD step)...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

# Create the training data
x, y = cPickle.load(open('OnlyNPs_codeonly.cPickle', 'rb'))
X_train = np.array(x, dtype='float32')
y_train = np.array(y, dtype='float32')

# Specify model and timing one SGD step
model = RNNTheano(10, 333, hidden_dim = 30)
#t1 = time.time()
#model.sgd_step(X_train[10], y_train[10], 0.005)
#t2 = time.time()
#print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

# Train model for n epoches
train_with_sgd(model, X_train, y_train, nepoch = 150, learning_rate = 0.01, evaluate_loss_after=1)

# Save model parameters
save_model_parameters_theano('model_parameters_OnlyNPs', model)

# Load previously saved model
#load_model_parameters_theano('outfile.npz', model)

