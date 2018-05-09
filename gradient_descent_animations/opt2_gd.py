'''
@name: opt2_gd.py
@description: Effects of the learning rate on the convergence of the gradient
              descent algorithm.
@author: VPi
@date: Jan 19 2018
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Define the function y = x^2 - 4x + 2
def func_y(x):
    y = x**2 - 4*x + 2

    return y

def gradient_descent(previous_x, learning_rate, epoch):
    x_gd = []
    y_gd = []
    
    x_gd.append(previous_x)
    y_gd.append(func_y(previous_x))

    # begin the loops to update x and y
    for i in range(epoch):
        current_x = previous_x - learning_rate*(2*previous_x - 4)
        x_gd.append(current_x)
        y_gd.append(func_y(current_x))

        # update previous_x
        previous_x = current_x

    return x_gd, y_gd
    
''' Plot our function '''
x = np.arange(-10, 14, 0.1)
y = func_y(x)

fig, ax = plt.subplots(2, 3, sharey = True, sharex = True)

for i in range(2):
    for j in range(3):
        ax[i,j].plot(x, y, lw = 0.9, color = 'k')
        ax[i,j].set_xlim([min(x), max(x)])
        ax[i,j].set_ylim([-10, max(y)+1])
        ax[i,j].set_xlabel(r'$x$')
        ax[i,j].set_ylabel(r'$y$')

''' Plot algorithm '''
# Initialize x0 and learning rate
x0 = -7
epoch = 25

learning_rate_1 = 0.03
learning_rate_2 = 0.4
learning_rate_3 = 1.02

x_gd_1, y_gd_1 = gradient_descent(x0, learning_rate_1, epoch)
x_gd_2, y_gd_2 = gradient_descent(x0, learning_rate_2, epoch)
x_gd_3, y_gd_3 = gradient_descent(x0, learning_rate_3, epoch)

# learning rate = 0.03
ax[0,0].scatter(x_gd_1, y_gd_1, c = 'b')
ax[0,0].text(0.02, 1.05, 'learning rate = 0.03 \n intial x = -7',
           transform=ax[0,0].transAxes)
for i in range(1, epoch+1):
    ax[0,0].annotate('', xy=(x_gd_1[i], y_gd_1[i]),
                   xytext=(x_gd_1[i-1], y_gd_1[i-1]),
                   arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                   va='center', ha='center')

# learning rate = 0.4
ax[0,1].scatter(x_gd_2, y_gd_2, c = 'b')
ax[0,1].text(0.02, 1.05, 'learning rate = 0.4 \n intial x = -7',
           transform=ax[0,1].transAxes)
for i in range(1, epoch+1):
    ax[0,1].annotate('', xy=(x_gd_2[i], y_gd_2[i]),
                   xytext=(x_gd_2[i-1], y_gd_2[i-1]),
                   arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                   va='center', ha='center')

# learning rate = 1.02
ax[0,2].scatter(x_gd_3, y_gd_3, c = 'b')
ax[0,2].text(0.02, 1.05, 'learning rate = 1.02 \n intial x = -7',
           transform=ax[0,2].transAxes)
for i in range(1, epoch+1):
    ax[0,2].annotate('', xy=(x_gd_3[i], y_gd_3[i]),
                   xytext=(x_gd_3[i-1], y_gd_3[i-1]),
                   arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                   va='center', ha='center')

'''
##################################
#            Animation           #
##################################
'''

# Create animation learning rate = 0.03
line1, = ax[1,0].plot([], [], 'r', label = 'Gradient descent', lw = 1.5)
point1, = ax[1,0].plot([], [], 'bo')

# Create animation learning rate = 0.4
line2, = ax[1,1].plot([], [], 'r', label = 'Gradient descent', lw = 1.5)
point2, = ax[1,1].plot([], [], 'bo')

# Create animation learning rate = 1.02
line3, = ax[1,2].plot([], [], 'r', label = 'Gradient descent', lw = 1.5)
point3, = ax[1,2].plot([], [], 'bo')

def init():
    line1.set_data([], [])
    point1.set_data([], [])

    line2.set_data([], [])
    point2.set_data([], [])

    line3.set_data([], [])
    point3.set_data([], [])

    return line1, point1, line2, point2, line3, point3,

def animate(i):
    # Animate line
    line1.set_data(x_gd_1[:i], y_gd_1[:i])    
    # Animate points
    point1.set_data(x_gd_1[i], y_gd_1[i])

    # Animate line
    line2.set_data(x_gd_2[:i], y_gd_2[:i])    
    # Animate points
    point2.set_data(x_gd_2[i], y_gd_2[i])

    # Animate line
    line3.set_data(x_gd_3[:i], y_gd_3[:i])    
    # Animate points
    point3.set_data(x_gd_3[i], y_gd_3[i])

    return line1, point1, line2, point2, line3, point3,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(x_gd_3), interval=120, 
                               repeat_delay=60, blit=True)

#anim.save('fig7.mp4', fps=10, extra_args=['-vcodec', 'libx264'])

plt.show()
