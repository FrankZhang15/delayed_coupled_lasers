# -*- coding: utf-8 -*-
"""
Created on Thu May 28 21:34:49 2020

@author: Yande
"""

import matplotlib.pyplot as plt
import numpy as npy
import scipy as sp
from pylab import linspace, array
from ddeint import ddeint


#%%
#Import Data (If necessary)
'''

Er1=sp.loadtxt('fig4_data/Er1.csv')
Ei1=sp.loadtxt('fig4_data/Ei1.csv')
Er2=sp.loadtxt('fig4_data/Er2.csv')
In1=sp.loadtxt('fig4_data/Intensity1.csv')
In2=sp.loadtxt('fig4_data/Intensity2.csv')
N1=sp.loadtxt('fig4_data/N1.csv')
N2=sp.loadtxt('fig4_data/N2.csv')
Ei2=sp.loadtxt('fig4_data/Ei2.csv')

'''



#%%
#Define Variables


T=392
alpha=2.6
P=0.23
tao_p=4.9E-12


Er10=0.4
Ei10=0
Er20=-0.39
Ei20=0
N10=0.05
N20=-0.05

tao=0.5
kappa=0.1
Cp=0.2*sp.pi

tt_minimum=0
tt_maximum=30000
tt_space=150000
#%%
#DDE Solver


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


g = lambda t: array([Er10,Ei10,Er20,Ei20,N10,N20])
tt = linspace(tt_minimum, tt_maximum, tt_space)


yy = ddeint(model, g, tt)


#%%
#Define Data and Save Data


Intensity1=yy[:, 0]**2+yy[:, 1]**2
Intensity2=yy[:, 2]**2+yy[:, 3]**2
N1=yy[:, 4]
N2=yy[:, 5]
Er1=yy[:, 0]
Ei1=yy[:, 1]
Er2=yy[:, 2]
Ei2=yy[:, 3]


E1=npy.vectorize(complex)(Er1,Ei1)
E2=npy.vectorize(complex)(Er2,Ei2)


npy.savetxt("Intensity1.csv", Intensity1, delimiter=",")
npy.savetxt("Intensity2.csv", Intensity2, delimiter=",")
npy.savetxt("N1.csv", N1, delimiter=",")
npy.savetxt("N2.csv", N2, delimiter=",")
npy.savetxt("Er1.csv", Er1, delimiter=",")
npy.savetxt("Ei1.csv", Ei1, delimiter=",")
npy.savetxt("Er2.csv", Er2, delimiter=",")
npy.savetxt("Ei2.csv", Ei2, delimiter=",")
npy.savetxt("E1.csv", E1, delimiter=",")
npy.savetxt("E2.csv", E2, delimiter=",")

t_axis=(sp.array(tt)*tao_p) *1E9

#%%
#Intensity Plotting


good_t=[138,140]
good_y_Intensity=[0,0.6]


plt.plot(t_axis, Intensity1, lw=1, color='orange', label="delay = %.01f" % tao)
plt.plot(t_axis, Intensity2, lw=1, color='blue', label="delay = %.01f" % tao)
plt.xlim(good_t)
plt.ylim(good_y_Intensity)
plt.ylabel("Intensity")
plt.xlabel('Time (ns)')
plt.title("Fig 3c\nTau= %.03f, k=%.3f, a=%.03f, T= %.01f, P= %.03f, Cp= %.03f" % (tao,kappa,alpha,T,P,Cp))
plt.legend(['Intensity1','Intensity2'],loc=1)
plt.savefig(f"Fig3c_Intensity.png",bbox_inches = 'tight')
plt.show()


#%%
#N plotting


good_y_N=[-0.06,0.06]


plt.plot(t_axis, N1, lw=1, color='orange', label="delay = %.01f" % tao)
plt.plot(t_axis, N2, lw=1, color='blue', label="delay = %.01f" % tao)
plt.xlim(good_t)
plt.ylim(good_y_N)
plt.xlabel("Time (ns)")
plt.ylabel("Inversion")
plt.legend(['N1','N2'],loc=1)
plt.title("Fig 3c\nTau= %.03f, k=%.3f, a=%.03f, T= %.01f, P= %.03f, Cp= %.03f" % (tao,kappa,alpha,T,P,Cp))
plt.savefig(f"Fig3c_Inversion.png",bbox_inches = 'tight')
plt.show()


#%%
#Fourier Part


t_Initial=int(20000/tt_maximum*tt_space)
t_End=int(29000/tt_maximum*tt_space)


t_Initial_s=t_Initial/tt_space*tt_maximum*tao_p
t_End_s=t_End/tt_space*tt_maximum*tao_p

#Getting the steady state t value


Initial_FourierE1=E1[t_Initial:t_End] 

#Initial_FourierE1_Ready=abs(Initial_FourierE1)**2


#Getting the steady state E value



FourierE1=npy.fft.fft(Initial_FourierE1, norm=None)

Plotting_FourierE1=abs(FourierE1)**2


#Fourier Transform for E1


Initial_FourierE2=E2[t_Initial:t_End]

#Initial_FourierE2_Ready=abs(Initial_FourierE2)**2


FourierE2=npy.fft.fft(Initial_FourierE2, norm=None)

Plotting_FourierE2=abs(FourierE2)**2


#Fourier Transform for E2



N=len(FourierE1)
Length_Array=[]
A=0
while A <= N-1:
    Length_Array.append(A)
    A+=1
Length_Array=npy.array(Length_Array)
Frequency=Length_Array*2*sp.pi/(t_End_s-t_Initial_s)


#Getting Frequency


Fourier_FinalE1=Plotting_FourierE1[1:]
Fourier_FinalE2=Plotting_FourierE2[1:]
Frequency_Final=Frequency[1:]


#Removing the "0" value



#%%
#Nyquist Frequency


Nfft=len(Frequency_Final)
Nyquist_Frequency=2*sp.pi/(tt_maximum/tt_space*tao_p)
Shift_Frequency=linspace(-Nyquist_Frequency/2, Nyquist_Frequency/2, Nfft)

Shift_Fourier_FinalE1=npy.fft.fftshift(Fourier_FinalE1)
Shift_Fourier_FinalE2=npy.fft.fftshift(Fourier_FinalE2)



#%%
#Fourier Plotting

plt.plot(Shift_Frequency, Shift_Fourier_FinalE1, lw=1, color='orange', label="delay = %.01f" % tao)
plt.xlim([-1E11,1E11])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Fourier_of_E1")
plt.legend(['Fourier_E1'],loc=1)
plt.title("Fig 3c\nTau= %.03f, k=%.3f, a=%.03f, T= %.01f, P= %.03f, Cp= %.03f" % (tao,kappa,alpha,T,P,Cp))
plt.savefig(f"Fig3c_FourierE1.png",bbox_inches = 'tight')

#%%
#Fourier Plotting

plt.plot(Shift_Frequency, Shift_Fourier_FinalE2, lw=1, color='blue', label="delay = %.01f" % tao)
plt.xlim([-1E11,1E11])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Fourier_of_E2")
plt.legend(['Fourier_E2'],loc=1)
plt.title("Fig 3c\nTau= %.03f, k=%.3f, a=%.03f, T= %.01f, P= %.03f, Cp= %.03f" % (tao,kappa,alpha,T,P,Cp))
plt.savefig(f"Fig3c_FourierE2.png",bbox_inches = 'tight')
#%%
#data save
npy.savetxt("Shift_Fourier_FinalE1.csv", Shift_Fourier_FinalE1, delimiter=",")
npy.savetxt("Shift_Fourier_FinalE2.csv", Shift_Fourier_FinalE2, delimiter=",")
npy.savetxt("Shift_Frequency.csv", Shift_Frequency, delimiter=",")





#%%
#Fourier Graph Analysis
X_dummyE1=[]

for i in Shift_Fourier_FinalE1:
    X_dummyE1.append(i)
    
#X_dummyE1.remove(max(X_dummyE1))
Fake_N=max(X_dummyE1)# finding 1stMax in X
Shift_Fourier_FinalE1_list=list(Shift_Fourier_FinalE1)
index=Shift_Fourier_FinalE1_list.index(Fake_N)+0#finding the position of 2ndMax in X
print(index)
print(Shift_Frequency[index])



