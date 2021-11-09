# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:27:09 2020

@author: Yande
"""
import scipy as sp
import numpy as npy
from scipy import integrate
from scipy.integrate import odeint
import matplotlib.pyplot as plt

T=392
alpha=2.6
P=0.23
t=npy.linspace(0,4000,80000)
x0=[0.4,0,0.05]


def equation(x,t):
    Er=x[0]
    Ei=x[1]
    N=x[2]
    dErdt = N*Er-alpha*N*Ei
    dEidt = N*Ei+alpha*N*Er
    dNdt = (P-N-(1+2*N)*(Ei**2+Er**2))/T
    return [dErdt, dEidt, dNdt]
print(equation(x0,0))


x = odeint(equation,x0,t)
Er = x[:,0]
Ei = x[:,1]
N = x[:,2]

plt.plot(t,Er,color="red")
plt.plot(t,Ei,color="blue")
plt.plot(t,N,color="black")
plt.legend(["Er","Ei","N"])
plt.show
     
#%%
plt.plot(t,Er**2+Ei**2)
plt.show
 #%%
opq=sp.sin(sp.pi)
print(opq)