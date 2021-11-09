# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:49:16 2020

@author: Yande
"""

import scipy as sp
import numpy as np
from scipy import integrate
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.special as special


T=392
alpha=2.6
P=0.23
t=npy.linspace(0,4000,80000)
x0=[0.4,0,0.05]

#%%
def equation(x,t):
    Er=x[0]
    Ei=x[1]
    N=x[2]
    dErdt = N*Er-alpha*N*Ei
    dEidt = N*Ei+alpha*N*Er
    dNdt = (P-N-(1+2*N)*(Ei**2+Er**2))/T
    return [dErdt, dEidt, dNdt]
print(equation(x0,0))

#%%
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
Int=Er**2+Ei**2
plt.plot(t,Int)
plt.show
 #%%
opq=sp.sin(sp.pi)
print(opq)
#%%


# Time domain representation of the resultant sine wave
plt.title('intensity')
plt.plot(t, Int)
plt.xlabel('Time')
plt.ylabel('intensity')
#%%
# Frequency domain representation
samplingFrequency=1/4000
fourierTransform = np.fft.fft(Int)/len(Int)           # Normalize amplitude
fourierTransform = fourierTransform[range(int(len(Int)/2))] # Exclude sampling frequency 
 
tpCount     = len(Int)
values      = np.arange(int(tpCount/2))
timePeriod  = tpCount/samplingFrequency
frequencies = values/timePeriod
 
# Frequency domain representation
plt.title('Fourier transform depicting the frequency components')
 
plt.plot(frequencies, abs(fourierTransform))
plt.xlabel('Frequency')
plt.ylabel('F(I)')
 
plt.show()
