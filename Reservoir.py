#!/usr/bin/env python3.6
# Reservoir.py
# Author: Shawn Beaulieu
# August 6th, 2018

import numpy as np
import tensorflow as tf

def Xavier_Initializer(name, shape):

    """
    To guard against both vanishing and exploding gradients. The variance
    of the distribution from which we draw random samples for weights
    is a function of the number of input neurons for a given layer
    (and for the case of Bengio initialization, the output neurons as well) 

    INPUTS:
    name: string containing exact name of layer being initialized
    shape: dimensions of the weight matrix: (e.g. (input,output)) 

    """
    return(tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True))


def Softmax(logits, temperature):
    return(tf.exp(logits/temperature)/tf.reduce_sum(tf.exp(logits/temperature), axis=1, keep_dims=True))

def Gaussian(y, mean, std):

    # Sample from Gaussian, where y is the target value during training    
    exponent = tf.mul(tf.sub(y, mean), tf.inv(std))
    exponent = -tf.square(sample)/2
    return(tf.mul(tf.exp(exponent), tf.inv(std)))*(1/(math.sqrt(2*math.pi)))

class MDNRNN():

    """
    Mixture Density Network RNN:
    
    No training is executed in this iteration of the MDNRNN. The network
    serves only as a memory reservoir, resulting in only minor degradations
    in performance as illustrated by https://ctallec.github.io/world-models/
    but boasting improvements in computational efficiency.

    """


    P = {

        'batch_size': 1,
        'seq_len': 1, # length of sequence passed
        'temperature': 1.0, # increase = higher uncertainty
        'num_mixtures': 5,
        'num_actions': 18,
        'latent_dim': 32,
        'rnn_size': 256,
        'use_layer_norm':True

    }

    def __init__(self, params={}):
 
        self.__dict__.update(MDNRNN.P, **params)
        # As per "World Models" (2018. Ha, Schmidhuber) we pass the encoded state
        # and the action space to the MDN-RNN (len(z) + len(action_space)).
        # But we only ouput the hypothesized latent encoding. Allow for variable sequences with 'None':       
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, self.num_actions+self.latent_dim])
        
        # Generate the skeleton of the untrained RNN:
        self.Build_RNN()
        self.Forward_Pass()

        # Start session and initialize variables
        self.sess = tf.Session()
        build = tf.global_variables_initializer()
        self.sess.run(build)

    def Build_RNN(self):

        self.rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell
        self.rnn_cell = self.rnn_cell(self.rnn_size, layer_norm=self.use_layer_norm)
        self.initial_state = self.rnn_cell.zero_state(self.batch_size, dtype=tf.float32)

        self.Parameterize()
    
    def Parameterize(self):

        # For each kth mixture, whose dimensions equal that of the latent dim,
        # we have 3 components corresponding to probability weight (pi),
        # and the mean (u) and standard deviation (s) for the kth mixture:

        output_dim = self.num_mixtures*self.latent_dim*3
        
        with tf.variable_scope("RNN"):
            self.W = Xavier_Initializer('outgoing_weights', shape=[self.rnn_size, output_dim])
            self.b = tf.get_variable('outgoing_biases', shape=[output_dim])

    def Forward_Pass(self):

        # Using the same settings as in World Models for the RNN:
        self.outputs, self.hidden_state = tf.nn.dynamic_rnn(self.rnn_cell, self.inputs, 
                        initial_state=self.initial_state, time_major=False, swap_memory=True, dtype=tf.float32, scope='RNN')

        # self.outputs at this point is just the output of the LSTM cell (size=rnn_size). Pass it through
        # the final layer by affine transformation with self.W (size=rnn_size X output_dim) to obtain
        # the last set of outputs for each mixture
        self.outputs = tf.reshape(self.outputs, [-1, self.rnn_size])
        self.outputs = tf.nn.xw_plus_b(self.outputs, self.W, self.b)
        self.outputs = tf.reshape(self.outputs, [-1, self.num_mixtures*3]) 

        self.Predict()

    def Predict(self):

        # Split the output tensor into 3 equivalently sized vectors. 
        # See Bishop (1994) for more details.    
        self.log_k_weights, self.log_k_means, self.k_stds = tf.split(self.outputs, 3, 1)

        k_weights = Softmax(self.log_k_weights, self.temperature)
        k_means = tf.exp(self.log_k_means)
        k_stds = self.k_stds

        """
            For each d in self.latent_dim, sample from mixture of k Gaussians.
            'gaussian_mixture' is a tensor containing "self.latent_dim" many vectors,
            each with k values.

        """

        #gaussian_mixture = Gaussian(tf.reshape(self.targets, [-1,1]), k_means, k_stds)

        # Scale-location transformation of unit normal: s*x + u
        
        gaussian_mixture = tf.random_normal(shape=tf.shape(k_stds), mean=0.0, stddev=1, dtype=tf.float32)

        # Element-wise multiplication of each row of the unit normal by a the standard
        # deviations for each mixture. Extend [k,1] vector using tf.stack(.)
        gaussian_mixture = tf.multiply(gaussian_mixture, k_stds)

        # Complete transformation of gaussian_mixture by adding the corresponding means:
        gaussian_mixture = tf.add(gaussian_mixture, k_means)

        """
            tf.shape(gaussian_mixture) = [self.latent_dim, self.num_mixtures]
            For each feature in self.targets, we're generating k Gaussian
            samples around that target feature (think how Gaussian(.) operates)

            Weight each vector by the corresponding weight obtained by MDN:
            shapes = [self.latent_dim, self.num_mixtures]*[self.num_mixtures, 1] 

            Take sum over axis=1, or across rows, each of which
            contains the weighted prediction for the corresponding
            target feature. 

            Resulting vector has size = [self.latent_dim,1]

        """ 

        self.prediction = tf.multiply(gaussian_mixture, k_weights)
        self.prediction = tf.reshape(tf.reduce_sum(self.prediction, axis=1, keep_dims=True), [1,-1])
         

    def Predict_Next_Frame(self, inputs, hidden_state):

        if hidden_state == None:
            hidden_state = self.sess.run(self.initial_state)

        self.seq_len = inputs.shape[0]
        feed = {self.inputs:inputs, self.initial_state:hidden_state}
        values = self.sess.run([self.prediction, self.hidden_state], feed_dict=feed)
        return(values)
