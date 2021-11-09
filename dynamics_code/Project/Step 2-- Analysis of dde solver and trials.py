# -*- coding: utf-8 -*-
"""
Created on Thu May 21 00:10:38 2020

@author: Yande
"""

import matplotlib.pyplot as plt
import numpy as npy
import scipy as sp
from pylab import linspace, array
from ddeint import ddeint



def model(Y, t):
    x, y,z = Y(t)
    xd, yd,zd = Y(t - 0.2)
    return array([xd,yd,z])


g = lambda t: array([1, 2,2])
tt = linspace(0, 3, 200)
yy1 = ddeint(model, g, tt)
plt.plot(tt, yy1[:, 2])
plt.legend("x","y")
plt.show()

#%%
def model(Y, t, d):
    x, y = Y(t)
    xd, yd = Y(t - d)
    return array([xd, yd])


g = lambda t: array([1, 2])
tt = linspace(0, 3, 200)


for d in [0, 0.2]:
    yy2 = ddeint(model, g, tt, fargs=(d,))
    plt.plot(tt, yy2[:, 1],color="red")
    plt.legend("x","y")
    plt.show()