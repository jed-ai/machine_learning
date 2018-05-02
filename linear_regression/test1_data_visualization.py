'''
Name: test1_data_visualization.py
Description: Visualization of the dataset "The World Happiness Report 2017"
Author: Phan Nguyen Vu
Date: 11st April 2018
'''

import csv
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

plt.figure(1)
#plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('The relationship between economy and happiness')
plt.xlabel('Economy (GDP per capita)')
plt.ylabel('Happiness score')
plt.plot(economy, happiness, 'x')
plt.show()

