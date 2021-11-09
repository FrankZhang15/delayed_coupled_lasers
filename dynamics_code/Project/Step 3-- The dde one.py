# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:12:25 2020

@author: Yande
"""


import numpy as npy
import scipy as sp
from scipy import integrate
from scipy.integrate import odeint
from ddeint import ddeint
import matplotlib.pyplot as plt


T=392
alpha=2.6
P=0.23
t=npy.linspace(0,4000,80000)
x0=[0.4,0,-0.4,0,0.05,0.05]



tao=0.5
kappa=0.1
Cp=0.1*sp.pi

def equation(X,t,tao):
    Er1, Ei1, Er2, Ei2, N1, N2 = X(t)
    Er1d, Ei1d, Er2d, Ei2d = X(t-tao)
    dEr1dt = N1*Er1-alpha*N1*Ei1+kappa*sp.cos(Cp)*Er2d+kappa*sp.sin(Cp)*Ei2d
    dEr2dt = alpha*N1*Er1+N1*Ei1+kappa*sp.cos(Cp)*Ei2d-kappa*sp.sin(Cp)*Ei2d
    dEi1dt = N2*Er2-alpha*N2*Ei2+kappa*sp.cos(Cp)*Er1d+kappa*sp.sin(Cp)*Ei1d
    dEi2dt = alpha*N2*Er2+N2*Ei2+kappa*sp.cos(Cp)*Ei1d-kappa*sp.sin(Cp)*Er1d
    dN1dt = (P-N1-(1+2*N1)*(Er1**2+Ei1**2))/T
    dN2dt = (P-N2-(1+2*N2)*(Er2**2+Ei2**2))/T
    return [dEr1dt, dEi1dt, dEr2dt, dEi2dt, dN1dt, dN2dt]
print(equation(x0,0,0.5))