import numpy as np

#Sample Activation Functions
def HidAct(x):
    '''An example function for the activation function of the hidden layers '''
    return np.tanh(x)

def OutAct(x):
    '''An example function for the activation function of the output layers '''
    return 1.0/(1.0+np.exp(-x))

#Sample derivative functions for error
def dHidAct(x):
    '''An example function for the derivative of the hidden activation.
    This is used in order to calculate the error to update the weights
    in back propagation '''
    return 1.0-x**2

def dOutAct(x):
    '''An example function for the derivative of the output activation.
    This is used in order to calculate the error to update the weights
    in back propagation '''
    return OutAct(x) * (1.0-OutAct(x))

class BackProp:
    def __init__(self, data, Input, Hidden, Output,LearningRate=0.1,Threshhold =0.5):
        '''Intakes the training data you want to use to train the system. Currently
        only works with differentiating two(2) classifications with labels 0 and 1.

        Intakes data as a dictionary with keys "vectors" and "labels" . Vectors should
        be organized as 2 lists, of x and y values, and labels as a single list of their corresponding
        labels.

        Input, Hidden and output all intake integers.
        Input sets the number of input layers you're giving the perceptron
        Hidden sets the number of hidden layers
        Output sets the number of output layers

        Learning Rate changes how quickly the algorithm converges, though a
        high learning rate will hurt accuracy
        '''

        self.X = data['vectors'][0]
        self.Y = data['vectors'][1]
        self.Labels = data['labels']
        self.LearningRate = LearningRate
        self.Threshhold = Threshhold

        self.Input = Input + 1 #Add 1 to act as a bias
        self.Hidden = Hidden
        self.Output = Output


        self.AInput = np.zeros(self.Input) + 1.0
        self.AHidden = np.zeros(self.Hidden) + 1.0
        self.AOutput = np.zeros(self.Output) + 1.0

        #Set random initial weights to be update through iterations
        self.WeightIn = np.random.rand(self.Input,self.Hidden)
        self.WeightOut = np.random.rand(self.Hidden,self.Output)

    def Response(self,Inputs):
        '''Returns the response the network has to the given input.
        Either 0 or 1 '''

        #Update inputs 0,1 as 2 stays constant at 1
        self.AInput[0],self.AInput[1] = Inputs[0],Inputs[1]

        #Hidden activation
        for i in range(self.Hidden):
            #Create a list of the inputs * their weight vector for each hidden node
            Temp = [self.AInput[_]*self.WeightIn[_][i] for _ in range(self.Input)]
            self.AHidden[i] = HidAct(sum(Temp))

        #Output Activation
        for i in range(self.Output):
            #Create a list of the inputs * their weight vector for each hidden node
            Temp = [self.AHidden[_]*self.WeightOut[_][i] for _ in range(self.Hidden)]
            self.AOutput[i] = OutAct(sum(Temp))

        return 1 if self.AOutput[0] > self.Threshhold else 0

    def UpdateWeight(self,Goal):
        '''Updates the weights and errors by back propagation based on
        the desired label(Goal) '''

        #Output Error
        OutError = [dOutAct(self.AOutput[_])*(Goal[_]-self.AOutput[_]) for _ in range(self.Output)]

        #Hidden Error
        HidError=[]
        for i in range(self.Hidden):
            Temp = [OutError[_]*self.WeightOut[i][_] for _ in range(self.Output)]
            HidError.append(dHidAct(self.AHidden[i])*sum(Temp))

        #Update Output Weight
        for i in range(self.Hidden):
            for k in range(self.Output):
                self.WeightOut[i][k] += self.LearningRate*OutError[k]*self.AHidden[i]

        #Update Input Weight
        for i in range(self.Input):
            for k in range(self.Hidden):
                self.WeightIn[i][k] += self.LearningRate*HidError[k]*self.AInput[i]


    def Learn(self,Iterations):
        '''Runs through the training data 'Iterations'-times in order to update the
        weights and train the network to correctly classify data. '''

        for i in range(Iterations):
            for _ in range(len(self.X)):            #Iterate the function
                self.Response([self.X[_],self.Y[_]]) #Update the variables
                self.UpdateWeight([self.Labels[_]]) #Learn the things


    def ClassificationList(self):
        '''Returns a list of the current classification of the training data.'''
        Results = []
        for _ in range(len(self.X)):
            self.Response([self.X[_],self.Y[_]])
            Results.append(1 if self.AOutput[0] > self.Threshhold else 0)
        return Results
