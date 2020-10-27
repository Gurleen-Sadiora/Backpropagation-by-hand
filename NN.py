import numpy as np
import pandas as pd
from numpy import savetxt, loadtxt

#Initialsing network
class NeuralNetwork():

    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.01):
        
        #Initialising weights of input layers and bias
        '''
        weights_input_hidden = np.random.normal(scale=1 / n_input ** .5,
                                        size=(n_input, n_hidden))

        weights_hidden_output = np.random.normal(scale=1 / n_input ** .5,
                                                 size=(n_hidden,n_output))

        bias_weights_input_hidden = np.random.normal(scale=1 / n_input ** .5,
                                                     size=(1,n_hidden))'''
                                                     
        bias_weights_hidden_output = np.random.normal(scale=1 / n_input ** .5,
                                                     size=(1,n_output))
        
        #Reloading saved weights so that we can train further, not starting from scratch
        weights_input_hidden = loadtxt('input_hidden02.csv', delimiter=',')
        weights_hidden_output = loadtxt('hidden_output02.csv', delimiter=',')
        bias_weights_input_hidden = loadtxt('bias_input_hidden02.csv', delimiter=',')
        #bias_weights_hidden_output = loadtxt('bias_hidden_output0.csv', delimiter=',')
        
        #Declaring model and its weights
        self.model = {'W1': weights_input_hidden, 'W2': weights_hidden_output, 'b1': bias_weights_input_hidden, 
                      'b2': bias_weights_hidden_output }
        
        self.n_input = n_input     #Input layer
        self.n_hidden = n_hidden   #Hidden layer
        self.n_output = n_output   #Output layer
        self.learning_rate = learning_rate    #Learning rate
    
        
        self.validation_error = -1   #Minimum validation and error so that it can be updated with the maximum error
        self.test_error = -1
             
    
    # Softmax function, making it stable so we don't get infinity values
    def softmax(self, x):
  
        log_c = np.max(x, axis=x.ndim - 1, keepdims=True)
        #for numerical stability
        y = np.sum(np.exp(x - log_c), axis=x.ndim - 1, keepdims=True)
        x = np.exp(x - log_c)/y
        #expA = np.exp(x)
        #return expA / expA.sum()
        return x
        
    
    
    #Model predicting function
    def predict(self, X):
        #Getting weights from model
        weights_input_hidden, bias_weights_input_hidden, weights_hidden_output, bias_weights_hidden_output =\
        self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        
        #Forward Pass
        hidden_input = np.dot(X, weights_input_hidden) + bias_weights_input_hidden
        hidden_output = np.tanh(hidden_input)
        output_input = np.dot(hidden_output, weights_hidden_output) + bias_weights_hidden_output
        output = self.softmax(output_input)#Using softmax function for multi-class prediction
        
        return output
        #return np.argmax(a2, axis=1)
        
    #Error computing function    
    def computeError(self, X, Y):
        
        weights_input_hidden, bias_weights_input_hidden, weights_hidden_output, bias_weights_hidden_output =\
        self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        
        hidden_input = np.dot(X, weights_input_hidden) + bias_weights_input_hidden
        hidden_output = np.tanh(hidden_input)
        output_input = np.dot(hidden_output, weights_hidden_output) + bias_weights_hidden_output
        output = self.softmax(output_input)
        predictions = np.argmax(output, axis=1)
        #print(type(predictions))
        #print(type(Y))
        return np.sum(predictions!=Y)/len(predictions) #Returning error
    
    
    #Loss computing function
    def computeLoss(self, X, Y):
        
        weights_input_hidden, bias_weights_input_hidden, weights_hidden_output, bias_weights_hidden_output =\
        self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        
        hidden_input = np.dot(X, weights_input_hidden) + bias_weights_input_hidden
        hidden_output = np.tanh(hidden_input)
        output_input = np.dot(hidden_output, weights_hidden_output) + bias_weights_hidden_output
        output = self.softmax(output_input)
        
        #Calculating cross entropy and data loss
        cross_entropy = -np.log(np.sum(Y * output, axis=1)+ 1e-12) 
    
        data_loss = np.sum(cross_entropy)
        
        # adding 0.001 regulatization term to loss, so that it does't get updated by a large number
        data_loss += 0.001/2 * (np.sum(np.square(weights_input_hidden)) + np.sum(np.square(weights_hidden_output)))
        
        return 1./X.shape[0] * data_loss
    
    
    
    #Training function
    def train(self, X_train, y_train, X_validate, y_validate,X_test, y_test, epochs=50, decay=0.001):
        
        weights_input_hidden, bias_weights_input_hidden, weights_hidden_output, bias_weights_hidden_output =\
        self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        
        for i in range(epochs):
            print("--> Epoch n°{} running...".format(i))
            # Forward propagation
            hidden_input = np.dot(X_train, weights_input_hidden) + bias_weights_input_hidden
            hidden_output = np.tanh(hidden_input)
            output_input = np.dot(hidden_output, weights_hidden_output) + bias_weights_hidden_output
            output = self.softmax(output_input)

            # Backpropagation
            
            #Change in cost wrt input given to output
            delta3 = output - y_train #Error
            ##Applying chain rule to find gradient descent for all parameters
            
            
            '''Baically, we need to minimize the cost function, and then update weights of input to hidden as well as hidden
            to output as cost decreases. To minimize the cost function, derivate of cost function is taken wrt to all weights
            of each layer.
            '''
            ##Change in weights of hidden to output
            dW2 = (hidden_output.T).dot(delta3)
            ##Change in bias of hidden to output
            db2 = np.sum(delta3, axis=0, keepdims = True)

            ##Derivate of hidden input weight wrt input
            delta2 = delta3.dot(weights_hidden_output .T) * (1 - np.power(hidden_output, 2))
            ##Change in weight and bias of input to hidden layer, since we are backpropagating the error
            #and updating weights
            dW1 = np.dot(X_train.T, delta2)
            db1 = np.sum(delta2, axis=0)  
            
            ##Updating weights and using regularization terms. This is done so that weights 
            #don't get updated by a large number. Bias don't have these terms
            dW2 += 0.01 * weights_hidden_output
            dW1 += 0.01 * weights_input_hidden

            #Updating learning rate and again using decay, so that it doesn't get updated by a large number
            self.learning_rate = self.learning_rate * 1/(1 + decay * i)

            # Gradient descent parameter update
            weights_input_hidden += -self.learning_rate * dW1
            bias_weights_input_hidden += -self.learning_rate * db1
            weights_hidden_output += -self.learning_rate * dW2
            bias_weights_hidden_output += -self.learning_rate * db2  

            ##Updating weights of the model
            self.model = {'W1': weights_input_hidden, 'W2': weights_hidden_output, 'b1': bias_weights_input_hidden, 
                      'b2': bias_weights_hidden_output }
            
            
            #Printing error after every 20 epochs
            if i % 20 == 0:
                print('-'*40)
                ##Calculating loss and error on training dataset
                loss = self.computeLoss(X_train, y_train)
                acc = self.computeError(X_train, y_train)
                ##Error on validation data
                error = self.computeError(X_validate, y_validate)
                print ("Loss after epoch {}: {}".format(i, loss ))
                print("Training error after epoch {}: {}%".format(i, error*100))
                print("Validation error after epoch {}: {}%".format(i, error*100))
                print('-'*40)                              
                

        print("Model trained!")
        self.trained = True
        validation_error = self.computeError(X_validate, y_validate)
        test_error = self.computeError(X_test, y_test)
        
        print("Validation error: {}%  |  Test error: {}%".format(validation_error*100, test_error*100))
        ##Updating with best validation errors and test errors
        if validation_error < np.abs(self.validation_error): #abs() for ini as it is -1
            print("New validation error!")
            self.validation_error = validation_error
        if test_error < np.abs(self.test_error):
            print("New test error!")
            self.test_error = test_error
            
    def resetTraining(self):
        self.trained = False
        self.validation_error = -1
        self.test_error = -1
        
        
##Helpful resources: https://github.com/EsterHlav/MLP-Numpy-Implementation-Gradient-Descent-Backpropagation/blob/master/NN.py
##https://stackabuse.com/creating-a-neural-network-from-scratch-in-python-multi-class-classification/
##https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    