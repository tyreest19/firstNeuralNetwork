import numpy as np
#scipy.special for the sigmoid function expit() 
import scipy.special

# neural network class definition
class neuralNetwork:

	#initialise the neural network
	def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
		# sets number of nodes in each layer
		self.inputNodes = inputNodes
		self.hiddenNodes = hiddenNodes
		self.outputNodes = outputNodes
		self.lr = learningRate
		#weights inside the arrays are w_i_j, where link is from node i to node j in the next layer 
		#w11 w21 #w12 w22 etc 
		self.wih = np.random.normal(0.0, pow(self.inputNodes, -0.5), (self.hiddenNodes, self.inputNodes)) 
		self.who = np.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.outputNodes, self.hiddenNodes))
		#activation function is the sigmoid function 
		self.activation_function = lambda x: scipy.special.expit(x)

	# train neural network
	def train (self, inputs_list, targets_list):
		#convert inputs list to 2d array 
		inputs = np.array(inputs_list, ndmin=2).T 
		targets = np.array(targets_list, ndmin=2).T 
		#calculate signals into hidden layer 
		hidden_inputs = np.dot(self.wih, inputs) 
		#calculate the signals emerging from hidden layer 
		hidden_outputs = self.activation_function(hidden_inputs) 
		#calculate signals into final output layer 
		final_inputs = np.dot(self.who, hidden_outputs) 
		#calculate the signals emerging from final output layer 
		final_outputs = self.activation_function(final_inputs)
		#error is the (target - actual)
		output_errors = targets - final_outputs
		hidden_errors = np.dot(self.who.T, output_errors)
		#update the weights for the links between the hidden and output layers  
		self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs)) 
		#update the weights for the links between the input and hidden layers 
		self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))


	# query neural network
	#query the neural network 
	def query(self, inputs_list):
		#convert inputs list to 2d array 
		inputs = np.array(inputs_list, ndmin=2).T 
		#calculate signals into hidden layer h
		hidden_inputs = np.dot(self.wih, inputs) 
		#calculate the signals emerging from hidden layer 
		hidden_outputs = self.activation_function(hidden_inputs) 
		#calculate signals into final output layer 
		final_inputs = np.dot(self.who, hidden_outputs) 
		#calculate the signals emerging from final output layer 
		final_outputs = self.activation_function(final_inputs) 
		return final_outputs



# number of input Nodes
inputNodes = 3
hiddenNodes = 3
outputNodes = 3 

# learning rate is 0.3
learning_rate = 0.3
n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learning_rate)
print(n.train([1.0, 0.75, -1.5], [0.5, 0.67, 0.56]))
print(n.query([1.0, 0.5, -1.5]))
