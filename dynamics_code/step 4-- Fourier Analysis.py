# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:01:25 2020

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



#tao=0.2
kappa=0.2
Cp=0.1*sp.pi


def model(X, t):
    Er1, Ei1, Er2, Ei2, N1, N2 = X(t)
    Er1d, Ei1d, Er2d, Ei2d, N1d, N2d = X(t-0.2)
    dEr1dt = N1*Er1-alpha*N1*Ei1+kappa*sp.cos(Cp)*Er2d+kappa*sp.sin(Cp)*Ei2d
    dEi1dt = alpha*N1*Er1+N1*Ei1+kappa*sp.cos(Cp)*Ei2d-kappa*sp.sin(Cp)*Er2d
    dEr2dt = N2*Er2-alpha*N2*Ei2+kappa*sp.cos(Cp)*Er1d+kappa*sp.sin(Cp)*Ei1d
    dEi2dt = alpha*N2*Er2+N2*Ei2+kappa*sp.cos(Cp)*Ei1d-kappa*sp.sin(Cp)*Er1d
    dN1dt = (P-N1-(1+2*N1)*(Er1**2+Ei1**2))/T
    dN2dt = (P-N2-(1+2*N2)*(Er2**2+Ei2**2))/T
    return [dEr1dt, dEi1dt, dEr2dt, dEi2dt, dN1dt, dN2dt]


g = lambda t: array([0.4,0,-0.4,0,0.05,-0.05]) # initial contitions at t =0
tt = linspace(0,30000, 150000)


yy = ddeint(model, g, tt)

sp.savetxt('DDE_solution.txt',yy)
#%%
intensity = sp.loadtxt('Intensity1.csv')













#%%


Intensity1=yy[:, 0]**2+yy[:, 1]**2
Intensity2=yy[:, 2]**2+yy[:, 3]**2
plt.plot(tt, Intensity1)
plt.plot(tt, Intensity2)
plt.ylabel("E")
plt.xlim([28000,28200])

plt.xlabel("t")
plt.ylabel("Intensity E1")
#plt.savefig("Intensity E1")


#%%


N1=yy[:, 4]
N2=yy[:, 5]
plt.plot(tt, yy[:, 4])
plt.plot(tt, yy[:, 5])
plt.xlim([48000,49000])


#%%

Er1=yy[:, 0]
Ei1=yy[:, 1]
Er2=yy[:, 2]
Ei2=yy[:, 3]


#%%
FEr1 =Er1[18600:19000]
FEi1 =Ei1[18600:19000]
FEr2 =Er2[18600:19000]
FEi2 =Ei2[18600:19000]

FFourierEr1=npy.fft.fft(FEr1, norm=None)
FFourierEi1=npy.fft.fft(FEi1, norm=None)
FFrequency1=(FFourierEr1**2+FFourierEi1**2)



FFrequency1_real=FFrequency1.real
FFrequency1_imag=FFrequency1.imag


FFourierEr2=npy.fft.fft(FEr2, norm=None)
FFourierEi2=npy.fft.fft(FEi2, norm=None)
FFrequency2=(FFourierEr2**2+FFourierEi2**2)


FFrequency2_real=FFrequency2.real
FFrequency2_imag=FFrequency2.imag

plt.plot(FFrequency1)
plt.plot(FFrequency2)
#plt.xlim([10,500])
#plt.savefig("Figure1_frequency1.png")


#%%
plt.plot(FEr2)

#%%
FourierEr1=npy.fft.fft(Er1, norm=None)
FourierEi1=npy.fft.fft(Ei1, norm=None)
Frequency1=(FourierEr1**2+FourierEi1**2)



Frequency1_real=Frequency1.real
Frequency1_imag=Frequency1.imag


FourierEr2=npy.fft.fft(Er2, norm=None)
FourierEi2=npy.fft.fft(Ei2, norm=None)
Frequency2=(FourierEr2**2+FourierEi2**2)


Frequency2_real=Frequency2.real
Frequency2_imag=Frequency2.imag

plt.plot(Frequency1)
plt.plot(Frequency2)
#plt.xlim([10,500])
#plt.savefig("Figure1_frequency1.png")





#%%


#plt.plot((Frequency1_real**2+Frequency1_imag**2)**(1/2))
plt.plot((Frequency2_real**2+Frequency2_imag**2)**(1/2))
#plt.xlim([0,2000])
plt.ylim([0,200000])
#plt.savefig("Figure1_frequency2.png")




