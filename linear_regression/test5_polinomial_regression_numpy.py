'''
Name: test5_polinomial_regression_numpy.py
Description: Find out the linear mode of the dataset
             "The World Happiness Report 2017" using Normal Equation.
Author: Phan Nguyen Vu
Date: 11st April 2018
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

# Order of polynomial
order = 9

# Converting X to the form X = [x1, x2, ..., xn] with xk = [k, xk]
def create_x(economy, order):
    # Create element matrix
    X = np.ones((len(economy), 1))
    for i in range(1, order+1):
        temp = np.power(economy, i)
        x_temp = np.array(temp, dtype=float)
        x_temp = np.transpose(np.matrix(x_temp))

        X = np.column_stack((X, x_temp))

    return X
X = create_x(economy, order)
# Converting y
y = np.array(happiness, dtype=float)
y = np.transpose(np.matrix(y))

# Using normal equation to find out the solution
params = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

# Plot model
line = []
line_x = np.arange(0,2,0.01)

params = params.tolist()

for i in range(len(line_x)):
    temp = params[0][0] # bias
    for j in range(1, order+1):
        temp = temp + params[j][0]*(line_x[i]**j)
    line.append(temp)

plt.figure(1)
#plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('The relationship between economy and happiness')
plt.xlabel('Economy (GDP per capita)')
plt.ylabel('Happiness score')
plt.plot(economy, happiness, 'x')
plt.plot(line_x, line)
plt.show()

