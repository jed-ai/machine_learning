'''
@name: gradient_calculation.py
@description: Calculating the gradient descent numerically and analytically
              using definition and calculus respectively.
@author: VPi
@date: 31 Jan 2018
'''

import numpy as np
import matplotlib.pyplot as plt

def func_y(x):
    y = x**2 - 4*x + 2

    return y

def grad_y(x):
    y = 2*x - 4

    return y

def tangent_y(x):
    y = -4*x + 2

    return y

def tangent_y2(x):
    y = func_y(4.5) + grad_y(4.5)*(x-4.5)

    return y

# y = x^2 - 4x + 2
x = np.arange(-1, 5, 0.01)
y = func_y(x)
y_t = tangent_y(x)
y_t2 = tangent_y2(x)

fig, ax = plt.subplots()

ax.plot(x, y, lw = 0.9, color = 'k')

# Left
ax.text(-0.3, 1.6, r'$ f(x) = f(0) +f\'(0)x $', rotation = -60)
ax.plot(x, y_t, lw = 0.9, linestyle = '--', color = 'r')
ax.scatter(0, 2, c='b')
ax.scatter(1.2, func_y(1.2), c='b')

ax.text(0.1, 2.3, r'$ \Delta x = - \alpha f\'(0) $')
ax.annotate('', xy=(1.2, 2), xytext=(0, 2),
                arrowprops={'arrowstyle': '->', 'color': 'c', 'lw': 1},
                va='center', ha='center')

ax.text(1.3, 0.3, r'$ \Delta y $', rotation = -90)
ax.annotate('', xy=(1.2, func_y(1.2)), xytext=(1.2, 2),
                arrowprops={'arrowstyle': '->', 'color': 'm', 'lw': 1},
                va='center', ha='center')

ax.annotate('', xy=(1.2, func_y(1.2)), xytext=(0, 2),
                arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                va='center', ha='center')

# Right
ax.text(3.3, 2.3, r'$ f(x) = f(4.5) +f\'(4.5)(x-4.5) $', rotation = 65)
ax.plot(x, y_t2, lw = 0.9, linestyle = '--', color = 'r')
ax.scatter(4.5, func_y(4.5), c='b')
ax.scatter(4.5 - 0.3*grad_y(4.5), func_y(4.5 - 0.3*grad_y(4.5)), c='b')

ax.text(4.5 - 0.3*grad_y(4.5)+0.3, func_y(4.5)+0.3, r'$ \Delta x = - \alpha f\'(4.5) $')
ax.annotate('', xy=(4.5 - 0.3*grad_y(4.5), func_y(4.5)),
                xytext=(4.5, func_y(4.5)),
                arrowprops={'arrowstyle': '->', 'color': 'c', 'lw': 1},
                va='center', ha='center')

ax.text(4.5 - 0.3*grad_y(4.5)-0.3, 1.5, r'$ \Delta y $', rotation = 90)
ax.annotate('', xy=(4.5 - 0.3*grad_y(4.5), func_y(4.5 - 0.3*grad_y(4.5))),
                xytext=(4.5 - 0.3*grad_y(4.5), func_y(4.5)),
                arrowprops={'arrowstyle': '->', 'color': 'm', 'lw': 1},
                va='center', ha='center')

ax.annotate('', xy=(4.5 - 0.3*grad_y(4.5), func_y(4.5 - 0.3*grad_y(4.5))),
                xytext=(4.5, func_y(4.5)),
                arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                va='center', ha='center')

ax.set_xlim([min(x), max(x)])
ax.set_ylim([-3, max(y)+1])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

plt.show()
