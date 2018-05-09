'''
@name: opt2_gd_3.py
@description: Effects of the initilization on the convergence of the gradient
              descent algorithm (non-convex function).
@author: VPi
@date: Jan 19 2018
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Define the function y = x^2 - 4x + 2
def func_y(x):
    y = x**2 + 8*np.cos(x) - 3*x

    return y

def gradient_descent(previous_x, learning_rate, epoch):
    x_gd = []
    y_gd = []
    
    x_gd.append(previous_x)
    y_gd.append(func_y(previous_x))

    # begin the loops to update x and y
    for i in range(epoch):
        current_x = previous_x - \
                    learning_rate*(2*previous_x -8*np.sin(previous_x) - 3)
        x_gd.append(current_x)
        y_gd.append(func_y(current_x))

        # update previous_x
        previous_x = current_x

    return x_gd, y_gd
    
''' Plot our function '''
x = np.arange(-2*np.pi, 2*np.pi, 0.1)
y = func_y(x)

fig, ax = plt.subplots(2, 2, sharey = True, sharex = True)

for i in range(2):
    for j in range(2):
        ax[i,j].plot(x, y, lw = 0.9, color = 'k')
        ax[i,j].set_xlim([min(x), max(x)])
        ax[i,j].set_ylim([-10, max(y)+1])
        ax[i,j].set_xlabel(r'$x$')
        ax[i,j].set_ylabel(r'$y$')

''' Plot algorithm '''
# Initialize x0 and learning rate
learning_rate_1 = 0.07
epoch = 15

x0_1 = -5
x0_2 = 5

x_gd_1, y_gd_1 = gradient_descent(x0_1, learning_rate_1, epoch)
x_gd_2, y_gd_2 = gradient_descent(x0_2, learning_rate_1, epoch)

# x0 = -5
ax[0,0].scatter(x_gd_1, y_gd_1, c = 'b')
ax[0,0].text(0.02, 1.05, 'learning rate = 0.07 \n intial x = -5',
           transform=ax[0,0].transAxes)
for i in range(1, epoch+1):
    ax[0,0].annotate('', xy=(x_gd_1[i], y_gd_1[i]),
                   xytext=(x_gd_1[i-1], y_gd_1[i-1]),
                   arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                   va='center', ha='center')

# x0 = 5
ax[0,1].scatter(x_gd_2, y_gd_2, c = 'b')
ax[0,1].text(0.02, 1.05, 'learning rate = 0.07 \n intial x = 5',
           transform=ax[0,1].transAxes)
for i in range(1, epoch+1):
    ax[0,1].annotate('', xy=(x_gd_2[i], y_gd_2[i]),
                   xytext=(x_gd_2[i-1], y_gd_2[i-1]),
                   arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                   va='center', ha='center')

'''
##################################
#            Animation           #
##################################
'''

# x0 = -5
line1, = ax[1,0].plot([], [], 'r', label = 'Gradient descent', lw = 1.5)
point1, = ax[1,0].plot([], [], 'bo')

# x0 = 5
line2, = ax[1,1].plot([], [], 'r', label = 'Gradient descent', lw = 1.5)
point2, = ax[1,1].plot([], [], 'bo')

def init():
    line1.set_data([], [])
    point1.set_data([], [])

    line2.set_data([], [])
    point2.set_data([], [])

    return line1, point1, line2, point2,

def animate(i):
    # Animate line
    line1.set_data(x_gd_1[:i], y_gd_1[:i])    
    # Animate points
    point1.set_data(x_gd_1[i], y_gd_1[i])

    # Animate line
    line2.set_data(x_gd_2[:i], y_gd_2[:i])    
    # Animate points
    point2.set_data(x_gd_2[i], y_gd_2[i])

    return line1, point1, line2, point2, 

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(x_gd_1), interval=120, 
                               repeat_delay=60, blit=True)

anim.save('fig7.mp4', fps=10, extra_args=['-vcodec', 'libx264'])

plt.show()
