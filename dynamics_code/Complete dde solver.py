# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:38:20 2020

@author: Yande
"""

import matplotlib.pyplot as plt
import numpy as npy
import scipy as sp
from pylab import linspace, array
from ddeint import ddeint

#%%
Ei1=sp.loadtxt('Ei1.csv')
Er1=sp.loadtxt('Er1.csv')
In1=sp.loadtxt('Intensity1.csv')
In2=sp.loadtxt('Intensity2.csv')


#%%
#Define Variables


T=392
alpha=2.6
P=0.23
tao_p=7.7E-12


Er10=0.4
Ei10=0
Er20=-0.4
Ei20=0
N10=0.05
N20=-0.05

tao=0.2
kappa=0.2
Cp=1.440

tt_minimum=0
tt_maximum=12000
tt_space=60000
#%%
#DDE Solver


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



#%%


g = lambda t: array([Er10,Ei10,Er20,Ei20,N10,N20])
tt = linspace(tt_minimum, tt_maximum, tt_space)



#%%
yy = ddeint(model, g, tt)


#%%
#Define Data and Save Data


Intensity1=yy[:, 0]**2+abs(yy[:, 1])**2
Intensity2=yy[:, 2]**2+yy[:, 3]**2
N1=yy[:, 4]
N2=yy[:, 5]
Er1=yy[:, 0]
Ei1=yy[:, 1]
Er2=yy[:, 2]
Ei2=yy[:, 3]



#%%

E1=Er1+1j*Ei1
E2=Er2+1j*Ei2


#%%
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
npy.savetxt("t_axis.csv", t_axis, delimiter=",")
#%%
t_axis=(sp.array(tt)*tao_p) *1E9
#%%
npy.savetxt("t_axis.csv", t_axis, delimiter=",")

#%%
#Intensity Plotting


good_t=[50,80]
good_y_Intensity=[-1,1]

params = {
   'axes.labelsize': 12,
   'font.size': 12,
   'legend.fontsize': 12,
   'xtick.labelsize': 16,
   'ytick.labelsize': 16,
   'figure.figsize': [8, 6]
   } 
plt.rcParams.update(params)
plt.plot(t_axis, Er1**2+Ei1**2, lw=1, color='orange', label="delay = %.01f" % tao)
#plt.plot(t_axis, Ei1, lw=1, color='blue', label="delay = %.01f" % tao)
plt.xlim(good_t)
#plt.ylim(good_y_Intensity)
plt.ylabel("Intensity")
plt.xlabel('Time (ns)')
plt.title("Intensity\nTau= %.03f, k=%.3f, a=%.03f, T= %.01f, P= %.03f, Cp= %.03f" % (tao,kappa,alpha,T,P,Cp))
plt.legend(['Er','Ei'],loc=1)
#plt.savefig(f"Intensity.png",bbox_inches = 'tight')
plt.show()


#%%
#N plotting


good_y_N=[-0.5,0]


plt.plot(t_axis, N1, lw=1, color='orange', label="delay = %.01f" % tao)
plt.plot(t_axis, N2, lw=1, color='blue', label="delay = %.01f" % tao)
plt.xlim(good_t)
#plt.ylim(good_y_N)
plt.xlabel("Time (ns)")
plt.ylabel("Inversion")
plt.legend(['N1','N2'],loc=1)
plt.title("Inversion\nTau= %.03f, k=%.3f, a=%.03f, T= %.01f, P= %.03f, Cp= %.03f" % (tao,kappa,alpha,T,P,Cp))
#plt.savefig(f"Inversion.png",bbox_inches = 'tight')
plt.show()

#%%
#Fourier Part


t_Initial=int(10000/tt_maximum*tt_space)
t_End=int(12000/tt_maximum*tt_space)


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

abs()
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

plt.plot(Frequency_Final, Shift_Fourier_FinalE1, lw=1, color='orange', label="delay = %.01f" % tao)
plt.xlim([15E11,25E11])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Fourier_of_E1")
plt.legend(['Fourier_E1'],loc=1)
plt.title("Fourier1\nTau= %.03f, k=%.3f, a=%.03f, T= %.01f, P= %.03f, Cp= %.03f" % (tao,kappa,alpha,T,P,Cp))
plt.savefig(f"FourierE1_noshift.png",bbox_inches = 'tight')

#%%
#Fourier Plotting

plt.plot(Shift_Frequency, Shift_Fourier_FinalE2, lw=1, color='blue', label="delay = %.01f" % tao)
plt.xlim([-2E11,2E11])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Fourier_of_E2")
plt.legend(['Fourier_E2'],loc=1)
plt.title("Fourier2\nTau= %.03f, k=%.3f, a=%.03f, T= %.01f, P= %.03f, Cp= %.03f" % (tao,kappa,alpha,T,P,Cp))
plt.savefig(f"FourierE2.png",bbox_inches = 'tight')
#%%
#data save
npy.savetxt("Shift_Fourier_FinalE1.csv", Shift_Fourier_FinalE1, delimiter=",")
npy.savetxt("Shift_Fourier_FinalE2.csv", Shift_Fourier_FinalE2, delimiter=",")
npy.savetxt("Shift_Frequency.csv", Shift_Frequency, delimiter=",")





#%%
#Fourier Graph Analysis

#%%
#Extra: Find Maximum
"""

X_dummyE1=[]

for i in Fourier_FinalE1[50:]:
    X_dummyE1.append(i)
    
#3X_dummyE1.remove(max(X_dummyE1))
Fake_N=max(X_dummyE1)# finding 3ndMax in X
Fourier_FinalE1_list=list(Fourier_FinalE1)
index=Fourier_FinalE1_list.index(Fake_N)#finding the position of 2ndMax in X
print(index)

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
    print(Shift_Fourier_FinalE1[index])
    print(Shift_Fourier_FinalE2[index])
    Alpha_F.append(Shift_Frequency[index])
    Alpha_Y1.append(Shift_Fourier_FinalE1[index])
    Alpha_Y2.append(Shift_Fourier_FinalE1[index])
"""