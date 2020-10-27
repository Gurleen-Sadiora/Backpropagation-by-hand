import numpy as np
import pandas as pd
import pickle

STUDENT_NAME = 'GURLEEN KAUR'
STUDENT_ID = '20848769'

def test_mlp(data_file):
	# Load the test set
	# START

	X = pd.read_csv(data_file, header = None)
	X = (X - X.mean()) / (X.std())
	X = X.fillna(0)
	
    # END


	# Load your network
	# START
	with open(r'.\clf6.pkl', 'rb') as input:
		NN = pickle.load(input)
	
	# END


	# Predict test set - one-hot encoded
	# y_pred = ...
	output = NN.predict(X)


	one_hot_predictions = np.empty((0,4), int)
	for i in range(len(output)):
		row = (output[i] == np.amax(output[i])).astype(float).reshape(1,4)
		one_hot_predictions = np.append(one_hot_predictions, row, axis=0)
		#print(one_hot_predictions)

	# return y_pred
	return one_hot_predictions

	

	

'''
How we will test your code:

from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 

y_pred = test_mlp('./test_data.csv')

test_labels = ...

test_accuracy = accuracy(test_labels, y_pred)*100
'''