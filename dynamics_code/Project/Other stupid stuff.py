# -*- coding: utf-8 -*-
"""
Created on Wed May 20 21:20:18 2020

@author: Yande
"""


import matplotlib.pyplot as plt
import numpy as npy
import scipy as sp
from pylab import linspace, array
from ddeint import ddeint


#%%
T=392
alpha=2.6
P=0.23



#tao=0.5
kappa=0.1
Cp=0.1*sp.pi


def model(X, t):
    Er1, Ei1, Er2, Ei2, N1, N2 = X(t)
    Er1d, Ei1d, Er2d, Ei2d, N1d, N2d = X(t-0.5)
    dEr1dt = N1*Er1-alpha*N1*Ei1+kappa*sp.cos(Cp)*Er2d+kappa*sp.sin(Cp)*Ei2d
    dEi1dt = alpha*N1*Er1+N1*Ei1+kappa*sp.cos(Cp)*Ei2d-kappa*sp.sin(Cp)*Er2d
    dEr2dt = N2*Er2-alpha*N2*Ei2+kappa*sp.cos(Cp)*Er1d+kappa*sp.sin(Cp)*Ei1d
    dEi2dt = alpha*N2*Er2+N2*Ei2+kappa*sp.cos(Cp)*Ei1d-kappa*sp.sin(Cp)*Er1d
    dN1dt = (P-N1-(1+2*N1)*(Er1**2+Ei1**2))/T
    dN2dt = (P-N2-(1+2*N2)*(Er2**2+Ei2**2))/T
    return [dEr1dt, dEi1dt, dEr2dt, dEi2dt, dN1dt, dN2dt]


g = lambda t: array([0.4,0,-0.4,0,0.05,0.05])
tt = linspace(0,10000, 100000)


yy = ddeint(model, g, tt)
#%%
Intensity1=yy[:, 0]**2+yy[:, 1]**2
plt.plot(tt, Intensity1)
plt.ylabel("E")
plt.xlim([0,10000])
plt.xlabel("t")
plt.ylabel("Intensity E1")
plt.savefig("Intensity E1")
#%%
plt.plot(tt, yy[:, 4])
plt.plot(tt, yy[:, 5])
