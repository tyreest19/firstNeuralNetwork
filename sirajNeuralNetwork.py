import numpy as np 

input_data = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
output_labels = np.array([[0, 1, 1, 0]]).T
print(input_data)
print(output_labels)

# Sigmoid function 
def activate(x, deriv=False):
	if deriv:
		return x*(1-x)
	return 1/(1 + np.exp(-x))

#weight matrix
synaptic_weights = 2 * np.random.random((3, 1)) - 1

activate(np.dot(input_data, synaptic_weights))
