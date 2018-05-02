'''
Name: test3_gradient_descent_numpy.py
Description: Find out the linear mode of the dataset
             "The World Happiness Report 2017" using Gradient Descent algorithm.
Author: Phan Nguyen Vu
Date: 2nd May 2018
'''

import csv
import numpy as np
import matplotlib.pyplot as plt

# Variables to store data
economy = []
happiness = []

# Read data from csv file
with open('world_happiness_report_2017.csv') as mycsvfile:
    datasets = csv.reader(mycsvfile, delimiter=',')
    i = 0
    for data in datasets:
        if i == 0:
            pass
        else:
            economy.append(float(data[5]))
            happiness.append(float(data[2]))
        i = i + 1

# Converting X to the form X = [x1, x2, ..., xn] with xk = [k, xk]
X = np.array(economy, dtype=float)
X = np.transpose(np.matrix(X))
element_matrix = np.ones((X.shape[0], 1))
X = np.column_stack((element_matrix, X))
# Converting y
y = np.array(happiness, dtype=float)
y = np.transpose(np.matrix(y))

# Using the gradient descent algorithm to find out the solution
# Initialize W
def initialization():
    W = np.random.randn(2, 1)
    print('The initialized W: ')
    print(W)
    return W 

# Compute gradient of E with respect to W
def compute_gradient(X, y, W):
    dW = (-1)*np.dot(X.T, y - np.dot(X, W))
    return dW 

# Gradient descent algorithm
def grads (X, y, iteration, learning_rate):
    cache = [] # Store the process of updating W
    W = initialization()
    cache.append([W[0, 0], W[1, 0]])

    for i in range(iteration):
        dW = compute_gradient(X, y, W)
        W = W - learning_rate*dW
        cache.append([W[0, 0], W[1, 0]])

    return W[0, 0], W[1, 0], cache

# Get the intercept and slope
intercept, slope, cache = grads(X, y, iteration = 25, learning_rate = 0.003)

# Store the process of updating W into a csv file
with open('updating_w.csv', 'w') as mycsvfile:
    for i in range(len(cache)):
        if i<(len(cache) - 1):
            mycsvfile.write(str(cache[i][0]) + ', ' + str(cache[i][1]) + ',')
        else:
            mycsvfile.write(str(cache[i][0]) + ', ' + str(cache[i][1]))

        mycsvfile.write('\n')

print('Intercept: ', intercept)
print('Slope: ', slope)

# Plot model
line = []
for data in economy:
    line.append(intercept + slope*data)

plt.figure(1)
#plt.xlim([0, 10])

plt.annotate('Linear model: y = ' +str(round(intercept, 2))+ ' + ' +str(round(slope, 2))+ 'x',
             xy = (economy[60], intercept + slope*economy[60]),
             xycoords='data', 
             xytext=(-170, +50), textcoords='offset points', fontsize=13,
             arrowprops=dict(arrowstyle="->"))

plt.ylim([0, 10])
plt.title('The relationship between economy and happiness')
plt.xlabel('Economy (GDP per capita)')
plt.ylabel('Happiness score')
plt.plot(economy, happiness, 'x')
plt.plot(economy, line)
plt.show()

