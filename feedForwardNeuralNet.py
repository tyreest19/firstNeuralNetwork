import math
import numpy as np 

class FeedForwardNeuralNetwork: 
	'''Single Layer Feed Forward Neural Network'''
	def __init__(self, inputNodes, learningRate):
		self.inputNodes = inputNodes
		self.outputNodes = 1
		self.learningRate = learningRate
		self.weights =  np.random.uniform(-1/math.sqrt(self.inputNodes), 1/math.sqrt(self.inputNodes),
											 size=self.inputNodes)

	def activationFunction(self, x):
		return 1/(1 + np.exp(-x))

	def train(self):
		pass

	
	def query(self, inputs):
		inputs = np.asarray(inputs)
		outputNodes = np.dot(self.weights, np.transpose(inputs))
		return self.activationFunction(outputNodes)

	

neuralNetwork = FeedForwardNeuralNetwork(3, 0.01)
print(neuralNetwork.query([0.2, 0.5, 0.6]))