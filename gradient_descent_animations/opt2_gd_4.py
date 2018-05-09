'''
@name: opt2_3d_gd.py
@description: Effects of the initilization on the convergence of the gradient
              descent algorithm (non-convex function).
              (3D version)
@author: VPi
@date: Jan 19 2018
'''

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

def func_z(x, y):
    # Calculate values of Z from the created grid
    z = (1 - 1/2.*x + x**5 +y**3)*np.exp(-x**2 - y**2)

    return z

def gradient_descent(previous_x, previous_y, learning_rate, epoch):
    x_gd = []
    y_gd = []
    z_gd = []

    x_gd.append(previous_x)
    y_gd.append(previous_y)
    z_gd.append(func_z(previous_x, previous_y))

    # begin the loops to update x, y and z
    for i in range(epoch):
        current_x = previous_x - learning_rate*\
                    ((-1/2. + 5.*previous_x**4)*np.exp(-previous_x**2-previous_y**2) + \
                     (1 - 1/2.*previous_x + previous_x**5 + previous_y**3)*np.exp(-previous_x**2-previous_y**2)*(-2*previous_x))
        x_gd.append(current_x)
        current_y = previous_y - learning_rate*\
                    (3*previous_y**2*np.exp(-previous_x**2-previous_y**2) +\
                     (1 - 1/2.*previous_x + previous_x**5 +previous_y**3)*np.exp(-previous_x**2-previous_y**2)*(-2*previous_y))
        y_gd.append(current_y)

        z_gd.append(func_z(current_x, current_y))

        # update previous_x and previous_y
        previous_x = current_x
        previous_y = current_y

    return x_gd, y_gd, z_gd

''' Plot our function '''
a = np.arange(-3, 3, 0.05)
b = np.arange(-3, 3, 0.05)
x, y = np.meshgrid(a, b)
z = func_z(x, y)

fig, ax = plt.subplots(2, 2, sharex = True, sharey = True)

for i in range(2):
    for j in range(2):
        ax[i,j].contour(x, y, z, cmap='jet')
        ax[i,j].set_xlabel(r'x')
        ax[i,j].set_ylabel(r'y')
        ax[i,j].set_xlim([-3, 3])
        ax[i,j].set_ylim([-3, 3])
        # Plot target (the minimum of the function)
        ax[i,j].plot(-1.7, 0., 'r*', markersize=10)

''' Plot the algorithm '''
learning_rate_1 = 0.3
epoch = 15

x_0_1 = -1
y_0_1 = 1
x_0_2 = 1
y_0_2 = -0.5

x_gd_1, y_gd_1, z_gd_1 = gradient_descent(x_0_1, y_0_1, learning_rate_1, epoch)

ax[0,0].plot(x_gd_1, y_gd_1, 'bo')
for i in range(1, epoch+1):
    ax[0,0].annotate('', xy=(x_gd_1[i], y_gd_1[i]),
                     xytext=(x_gd_1[i-1], y_gd_1[i-1]),
                   arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                   va='center', ha='center')
ax[0,0].text(0.02, 1.05, 'learning rate = 0.3 \n'+r'$x_{0}=-1$' +'\n' + \
             r'$y_{0}=1$', transform=ax[0,0].transAxes)

x_gd_2, y_gd_2, z_gd_2 = gradient_descent(x_0_2, y_0_2, learning_rate_1, epoch)

ax[0,1].plot(x_gd_2, y_gd_2, 'bo')
for i in range(1, epoch+1):
    ax[0,1].annotate('', xy=(x_gd_2[i], y_gd_2[i]),
                     xytext=(x_gd_2[i-1], y_gd_2[i-1]),
                   arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                   va='center', ha='center')
ax[0,1].text(0.02, 1.05, 'learning rate = 0.3 \n'+r'$x_{0}=1$' +'\n' + \
             r'$y_{0}=-0.5$', transform=ax[0,1].transAxes)

# alpha = 0.7
line1, = ax[1,0].plot([], [], 'r', label = 'Gradient descent', lw = 1.5)
point1, = ax[1,0].plot([], [], 'bo')

# alpha = 0.3
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





