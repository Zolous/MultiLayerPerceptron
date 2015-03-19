import numpy as np
import matplotlib.pyplot as plt
import pylab
from TestData import Data
from MultilayerPerceptron import BackProp


#Sample Activation Functions in 'Multilayer Perceptron' were used.
Example = BackProp(Data,2,10,1,LearningRate=0.5)

#Create a meshgrid to pass through the class in order to create a contour plot
y = np.arange(-8.01, 8.01, 0.25)
x = np.arange(-8.01, 8.01, 0.25)
X,Y = pylab.meshgrid(x,y)
Z = np.zeros([(len(X)),len(X[0])])

for i in range(len(Z)):
    for j in range(len(Z[0])):
        Z[i][j] = Example.Response([X[i][j],Y[i][j]])

plt.xlim(xmin=-8,xmax=8)
plt.ylim(ymin=-8,ymax=8)
plt.contourf(X, Y, Z, levels=[0,1], extend='both')
plt.scatter(Example.X,Example.Y,c=['w' if _ == 0 else 'k' for _ in Example.Labels],label='Training Data')
plt.title('Contour plot with training data before training')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('Untrained.png')
plt.clf()

Example.Learn(20)
for i in range(len(Z)):
    for j in range(len(Z[0])):
        Z[i][j] = Example.Response([X[i][j],Y[i][j]])

plt.xlim(xmin=-8,xmax=8)
plt.ylim(ymin=-8,ymax=8)
plt.contourf(X, Y, Z, levels=[0,1], extend='both')
plt.scatter(Example.X,Example.Y,c=['w' if _ == 0 else 'k' for _ in Example.ClassificationList()],label='Training Data')
plt.title('Contour plot with training data after training')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('Trained.png')
plt.clf()