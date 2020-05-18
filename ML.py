import numpy as np
from scipy import optimize

#training values
# X = (business hours, times sanitizer unit has detected a person), y = number of times sanitizer is dispensed
X = np.array(([9,32], [10,46], [11,52], [12,90], [13,104], [14,28], [15,36], [16,82], [17,20]), dtype=float)
y = np.array(([28], [36], [50], [81], [97], [26], [31], [71], [14]), dtype=float)

# Normalisation
X = X/np.amax(X, axis=1)
y = y/(np.amax(X, axis=1) * 2) # maximum value for sanitizer dispenses is twice the number of people detected to allow for users dispensing more than once

class NeuralNetwork(object):

    def __init__(self):
        #Define hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        #Define weight parameters
        self.weight1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.weight2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        #Propagate inputs through network
        self.z2 = np.dot(X, self.weight1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.weight2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        #Apply sigmoid activation function to a scalar, vector or matrix
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        #Derivative of Sigmoid Function
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunction(self, X, y):
        #Compute cost for X and y using the weights already stored in the class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
    
    def costFunctionPrime(self, X, y):
        #Compute the derivative with respect to weight1 and weight2
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        derivativeJweight2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.weight2.T)*self.sigmoidPrime(self.z2)
        derivativeJweight1 = np.dot(X.T, delta2)

        return derivativeJweight1, derivativeJweight2

    #Functions to make it easier to interact with other classes
    def getParameters(self):
        parameters = np.concatenate((self.weight1.ravel(), self.weight2.ravel())) #gets weight1 and weight2 as vectors
        return parameters

    def setParameters(self, parameters):
        #sets weight1 and weight2 using single parameter vectors
        weight1Start = 0
        weight1End = self.hiddenLayerSize * self.inputLayerSize
        self.weight1 = np.reshape(parameters[weight1Start:weight1End], (self.inputLayerSize, self.hiddenLayerSize))
        weight2End = weight1End + self.hiddenLayerSize*self.outputLayerSize
        self.weight2 = np.reshape(parameters[weight1End:weight2End], self.hiddenLayerSize, self.outputLayerSize)

    def computeGradients(self, X, y):
        derivativeJweight1, derivativeJweight2 = self.costFunctionPrime(X, y)
        return np.concatenate((derivativeJweight1.ravel(), derivativeJweight2.ravel()))

def computeNumericalGradient(N, X, y):
    parametersInitial = N.getParameters()
    numgrad = np.zeros(parametersInitial.shape)
    perturb = np.zeros(parametersInitial.shape)

    e = 1e-4

    for p in range(len(parametersInitial)):
        #Set the perturbation vector
        perturb[p] = e
        N.setParameters(parametersInitial + perturb)
        loss2 = N.costFunction(X, y)

        N.setParameters(parametersInitial - perturb)
        loss1 = N.costFunction(X, y)

        #Compute the Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)

        #return the value that was changed to zero
        perturb[p] = 0

    #return parameters to original value
    N.setParameters(parametersInitial)

    return numgrad

class trainer(object):

    def __init__(self, NN):
        self.N = NN

    def costFunctionWrapper(self, parameters, X, y):
        self.N.setParameters(parameters)
        cost = self.N.costFunction(X, y)
        gradient = self.N.computeGradients(X, y)
        return cost, gradient

    def callBackF(self, parameters):
        self.N.setParameters(parameters)
        self.J.append(self.N.costFunction(self.X, self.y))


    def train(self, X, y):
        self.X = X
        self.y = y

        self.J = []

        parameters0 = self.N.getParameters()

        options = {'maxiter':200, 'disp':True}
        _res = optimize.minimize(self.costFunctionWrapper, parameters0, jac=True, method='BFGS', args=(X,y), options=options, callback=self.callBackF)

        self.N.setParameters(_res.x)
        self.optimizationResults = _res