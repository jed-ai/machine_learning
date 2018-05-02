'''
Name: test4_w__updating_animation.py
Description: -   Animate the process of updating weights to find the linear model
             for "The World Happiness Report 2017" using Gradient Descent
             algorithm.
             -   Need to run test3_gradient_descent_numpy.py first to create the
             csv file that stores updating weights
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

# Variale to store updating weights
weights = []
# Read weights from csv file
with open('updating_w.csv') as mycsvfile:
    temps = csv.reader(mycsvfile, delimiter=',')
    for temp in temps:
        weights.append([float(temp[0]), float(temp[1])])

# Converting X to the form X = [x1, x2, ..., xn] with xk = [k, xk]
X = np.array(economy, dtype=float)
X = np.transpose(np.matrix(X))
element_matrix = np.ones((X.shape[0], 1))
X = np.column_stack((element_matrix, X))
# Converting y
y = np.array(happiness, dtype=float)
y = np.transpose(np.matrix(y))

print(weights)

'''
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
'''
