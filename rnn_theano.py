import numpy as np
import theano as theano
import theano.tensor as T
from utils import *
theano.config.floatX = 'float32'
class RNNTheano:
    
    def __init__(self, input_dim, word_dim, hidden_dim=100, bptt_truncate=5):
        # input_dim: input dimension
        # word_dim: output dimension (clases or vocabulary)
        # hidden_dim: hidden dimension
        
        # Assign instance variables
        self.input_dim = input_dim
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        U = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (hidden_dim, input_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))      
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
    
    def __theano_build__(self):
        U, V, W = self.U, self.V, self.W
        x = T.matrix('x')
        y = T.matrix('y')
        def forward_prop_step(x_t, s_t_prev, U, V, W):
            s_t = T.tanh(U.dot(x_t) + W.dot(s_t_prev)) # modified for compatibility with a vector input instead of a index
            o_t = T.nnet.softmax(V.dot(s_t))
            return [o_t[0], s_t]
            # this is the loop for fordward calculation of outputs for words throuh the the sentence array x (each sentence has multiple vectors representing words...)
        [o,s], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[U, V, W],
            truncate_gradient=self.bptt_truncate,
            strict=True)
        
        prediction = T.argmax(o, axis=1)
        true_class = T.argmax(y, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        accuracy = T.sum(T.eq(prediction, true_class))
        
        # Gradients
        dU = T.grad(o_error, U)
        dV = T.grad(o_error, V)
        dW = T.grad(o_error, W)
        
        # Assign functions
        self.forward_propagation = theano.function([x], [o, s])
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [dU, dV, dW])
        self.totaccuracy = theano.function([x, y], accuracy)
        
        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x,y,learning_rate], [], 
                      updates=[(self.U, self.U - learning_rate * dU),
                              (self.V, self.V - learning_rate * dV),
                              (self.W, self.W - learning_rate * dW)], allow_input_downcast=True)
    
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)
    
    def calculate_accuracy(self, X, Y):
        return np.sum([self.totaccuracy(x,y) for x,y in zip(X,Y)])

