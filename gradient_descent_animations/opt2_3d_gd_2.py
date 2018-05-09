'''
@name: opt2_3d_gd_2.py
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

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(x, y, z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')

# Plot target (the minimum of the function)
min_point = np.array([-1.7, 0.])
min_point_ = min_point[:, np.newaxis]
ax.plot(*min_point_, func_z(*min_point_), 'r*', markersize=10)

''' Animation '''
# Initialize x0 and learning rate
x_0_1 = -2.9
y_0_1 = -0.3
x_0_2 = 1
y_0_2 = -0.5
learning_rate = 0.3
epoch = 15

x_gd_1, y_gd_1, z_gd_1 = gradient_descent(x_0_1, y_0_1, learning_rate, epoch)
x_gd_2, y_gd_2, z_gd_2 = gradient_descent(x_0_2, y_0_2, learning_rate, epoch)

# alpha = 0.7
line1, = ax.plot([], [], 'r', label = 'Gradient descent', lw = 1.5)
point1, = ax.plot([], [], 'bo')

# alpha = 0.3
line2, = ax.plot([], [], 'r', label = 'Gradient descent', lw = 1.5)
point2, = ax.plot([], [], 'bo')

def init():
    line1.set_data([], [])
    line1.set_3d_properties([])
    point1.set_data([], [])
    point1.set_3d_properties([])

    line2.set_data([], [])
    line2.set_3d_properties([])
    point2.set_data([], [])
    point2.set_3d_properties([])

    return line1, point1, line2, point2,

def animate(i):
    # Animate line
    line1.set_data(x_gd_1[:i], y_gd_1[:i])
    line1.set_3d_properties(z_gd_1[:i])
    # Animate points
    point1.set_data(x_gd_1[i], y_gd_1[i])
    point1.set_3d_properties(z_gd_1[i])

    # Animate line
    line2.set_data(x_gd_2[:i], y_gd_2[:i])
    line2.set_3d_properties(z_gd_2[:i])
    # Animate points
    point2.set_data(x_gd_2[i], y_gd_2[i])
    point2.set_3d_properties(z_gd_2[i])

    return line1, point1, line2, point2, 

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(x_gd_1), interval=120, 
                               repeat_delay=60, blit=True)

anim.save('fig7.mp4', fps=10, extra_args=['-vcodec', 'libx264'])

plt.show()





