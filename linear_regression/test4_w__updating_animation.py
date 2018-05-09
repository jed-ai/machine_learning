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
from matplotlib import animation

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

# Calculate some coefficients
economy = np.array(economy, dtype = float)
happiness = np.array(happiness, dtype = float)
y = np.sum(happiness)
x = np.sum(economy)
y_sqr = np.sum(happiness**2)
x_sqr = np.sum(economy**2)
xy = np.sum(happiness*economy)
N = len(happiness)

# Optimal point
w1_opt = (xy - 1/N*y*x)/(x_sqr - 1/N*x*x)
w0_opt = 1/N*(y - w1_opt*x)
print('The optimal values of W0 and W1, respectively')
print([w0_opt, w1_opt])

# Plot cost function
def cost_func(w0, w1):
    global x, y, x_sqr, y_sqr, xy, N
    z = y_sqr + x_sqr*(w1**2) + N*(w0**2) - 2*xy*w1 - 2*y*w0 + 2*x*w0*w1
    return z

a = np.arange(-10, 10, 0.05)
b = np.arange(-10, 10, 0.05)
w0, w1 = np.meshgrid(a, b)
z = cost_func(w0, w1)

fig, ax = plt.subplots(1, 2, figsize = (12, 5))
ax[0].plot(economy, happiness, 'x')
ax[0].set_xlabel('Economy')
ax[0].set_ylabel('Hapiness')

ax[1].contour(w0, w1, z, 50)
ax[1].plot(w0_opt, w1_opt, 'r*', markersize=10)
ax[1].set_xlabel('w0')
ax[1].set_ylabel('w1')

# Create lists storing the updating w0 and w1
w0_grad = []
w1_grad = []
for w in weights:
    w0_grad.append(w[0])
    w1_grad.append(w[1])

# Create list storing the updating model
models = []
model = []

for w in weights:
    model = []
    for data in economy:
        model.append(w[0] + w[1]*data)
    models.append(model)


# Animation
# Gradient update
line1, = ax[1].plot([], [], 'r', label='GD', lw = 1.5)
point1, = ax[1].plot([], [], 'bo')
cost_display = ax[1].text(-1, 3.2, '')

# Model update
line2, = ax[0].plot([], [], lw = 0.9)
weights_display = ax[0].text(0.05, .9, '', transform=ax[0].transAxes)

def init():
    line1.set_data([], [])
    point1.set_data([], [])
    cost_display.set_text('')

    line2.set_data([], [])
    weights_display.set_text('')

    return line1, point1, line2, cost_display, weights_display

def animate(i):
    line1.set_data(w0_grad[:i], w1_grad[:i])
    point1.set_data(w0_grad[i], w1_grad[i])
    cost_display.set_text('Cost = ' + str(cost_func(w0_grad[i], w1_grad[i])) +\
                          '\nIteration: ' + str(i))

    line2.set_data(economy, models[i])
    weights_display.set_text('Our model: \n' + \
                             'y = ' + str(round(w0_grad[i], 2)) + ' + ' + \
                             str(round(w1_grad[i], 2)) + '*x')

    return line1, point1, line2,cost_display, weights_display

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(w0_grad), interval=120, 
                               repeat_delay=60, blit=True)

anim.save('gd_animation.mp4', fps=10, extra_args=['-vcodec', 'libx264'])

plt.show()

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
