'''
Name: test2_normal_equation_numpy.py
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

# Converting X to the form X = [x1, x2, ..., xn] with xk = [k, xk]
X = np.array(economy, dtype=float)
X = np.transpose(np.matrix(X))
element_matrix = np.ones((X.shape[0], 1))
X = np.column_stack((element_matrix, X))
# Converting y
y = np.array(happiness, dtype=float)
y = np.transpose(np.matrix(y))

# Using normal equation to find out the solution
weights = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
intercept = weights[0, 0]
slope = weights[1, 0]

print('Intercept: ', intercept)
print('Slope: ', slope)

# Plot model
line = []
for data in economy:
    line.append(intercept + slope*data)

plt.figure(1)
#plt.xlim([0, 10])

plt.annotate(r'Linear model: $y = 3.2 + 2.2x$',
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

