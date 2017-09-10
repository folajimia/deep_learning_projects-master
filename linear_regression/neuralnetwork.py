
# import required libraries
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt



class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                        (self.hidden_nodes, self.input_nodes))

        # print (self.weights_input_to_hidden.shape) #(6,56)


        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes ** -0.5,
                                                         (self.output_nodes, self.hidden_nodes))

        # print (self.weights_hidden_to_output.shape) #(6,56)

        self.lr = learning_rate

        #### Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array

        print(inputs_list.shape)
        print(targets_list.shape)

        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # print(inputs.shape)#(56,1)
        # print(targets.shape)#(1,1)

        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)  # signals into hidden layer
        # print(hidden_inputs.shape) (6,1)
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer
        # print(hidden_outputs.shape) (6,1)
        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)  # signals into final output layer
        # (6,1)(1,6)
        # print(final_inputs.shape)
        final_outputs = final_inputs  # signals from final output layer
        # print(final_outputs.shape)

        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error
        # error = y-y'= targets - final_outputs
        output_errors = (targets - final_outputs) * 1  # Output layer error is the difference between desired target and actual output since outer layer input equals output.
        # print (output_errors.shape)

        # TODO: Backpropagated error
        # error
        # error for the hidden unit with backpropagation=np.dot(self.weights_hidden_to_output.T, output_errors) * (hidden_outputs*(1-hidden_outputs))

        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors)  # errors propagated to the hidden layer
        hidden_grad = hidden_outputs * (1 - hidden_outputs)  # hidden layer gradients

        # TODO: Update the weights
        self.weights_hidden_to_output += self.lr * output_errors * hidden_outputs.T  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * hidden_errors * hidden_grad * inputs.T  # update input-to-hidden weights with gradient descent step

    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        #### Implement the forward pass here ####
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer

        return final_outputs
