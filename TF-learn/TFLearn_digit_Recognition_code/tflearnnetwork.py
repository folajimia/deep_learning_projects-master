import numpy as np
import tensorflow as tf
import tflearn
import dataprep


# Define the neural network
def build_model():
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()
    net = tflearn.input_data([None, dataprep.trainX.shape[1]])

    # Hidden layer
    #net = tflearn.fully_connected(net, 784, activation='ReLU')
    #net = tflearn.fully_connected(net, 500, activation='ReLU')
    net = tflearn.fully_connected(net, 200, activation='ReLU')
    net = tflearn.fully_connected(net, 25, activation='ReLU')

    net = tflearn.fully_connected(net, 10, activation='softmax')

    # output layer
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')

    #### Your code ####
    # Include the input layer, hidden layer(s), and set how you want to train the model

    # This model assumes that your network is named "net"
    model = tflearn.DNN(net)
    return model