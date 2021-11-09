# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:57:19 2020

@author: Yande
"""

import matplotlib.pyplot as plt
import numpy as npy
import scipy as sp
from pylab import linspace, array
from ddeint import ddeint

#%%
#Define Variables


T=392
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
Cp=0.1*sp.pi

tt_minimum=0
tt_maximum=6000
tt_space=30000
E1_peak=[]
E2_peak=[]
Freq_1=[]
Freq_2=[]

#%%
'''Changing Alpha'''
#While loop generates an array of alpha:
alpha=2.4
limit=2.8
alphas=[]
while alpha < limit:
    alphas.append(alpha)
    alpha += 0.1
print(alphas)
#%%
#creat FOR loop for codes:

for alpha in alphas:
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

    

    


    g = lambda t: array([Er10,Ei10,Er20,Ei20,N10,N20])
    tt = linspace(tt_minimum, tt_maximum, tt_space)


    yy = ddeint(model, g, tt)



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



    npy.savetxt("Intensity1_"+str(alpha)+".csv", Intensity1, delimiter=",")
    npy.savetxt("Intensity2_"+str(alpha)+".csv", Intensity2, delimiter=",")
    npy.savetxt("N1_"+str(alpha)+".csv", N1, delimiter=",")
    npy.savetxt("N2_"+str(alpha)+".csv", N2, delimiter=",")
    npy.savetxt("Er1_"+str(alpha)+".csv", Er1, delimiter=",")
    npy.savetxt("Ei1_"+str(alpha)+".csv", Ei1, delimiter=",")
    npy.savetxt("Er2_"+str(alpha)+".csv", Er2, delimiter=",")
    npy.savetxt("Ei2_"+str(alpha)+".csv", Ei2, delimiter=",")
    npy.savetxt("E1_"+str(alpha)+".csv", E1, delimiter=",")
    npy.savetxt("E2_"+str(alpha)+".csv", E2, delimiter=",")

    t_axis=(sp.array(tt)*tao_p) *1E9


#Intensity Plotting



#Fourier Part


    t_Initial=int(4000/tt_maximum*tt_space)
    t_End=int(6000/tt_maximum*tt_space)


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



    n=len(FourierE1)
    N=n-1
    Length_Array=[]
    A=0
    while A <= N:
        Length_Array.append(A)
        A+=1
    Length_Array=npy.array(Length_Array)
    Frequency=Length_Array*2*sp.pi/(t_End_s-t_Initial_s)


#Getting Frequency


    Fourier_FinalE1=Plotting_FourierE1[1:]
    Fourier_FinalE2=Plotting_FourierE2[1:]
    Frequency_Final=Frequency[1:]


#Removing the "0" value



#Nyquist Frequency


    Nfft=len(Frequency_Final)
    Nyquist_Frequency=2*sp.pi/(tt_maximum/tt_space*tao_p)
    Shift_Frequency=linspace(-Nyquist_Frequency/2, Nyquist_Frequency/2, Nfft)

    Shift_Fourier_FinalE1=npy.fft.fftshift(Fourier_FinalE1)
    Shift_Fourier_FinalE2=npy.fft.fftshift(Fourier_FinalE2)



    
#data save
    npy.savetxt("Shift_Fourier_FinalE1_"+str(alpha)+".csv", Shift_Fourier_FinalE1, delimiter=",")
    npy.savetxt("Shift_Fourier_FinalE2_"+str(alpha)+".csv", Shift_Fourier_FinalE2, delimiter=",")
    npy.savetxt("Shift_Frequency_"+str(alpha)+".csv", Shift_Frequency, delimiter=",")






    Peak1=max(Shift_Fourier_FinalE1)
    Shift_Fourier_FinalE1_list=list(Shift_Fourier_FinalE1)
    index1=Shift_Fourier_FinalE1_list.index(Peak1)
    value_before_index1=index1-1
    freq1=Shift_Frequency[value_before_index1:index1]
    print(freq1)
    print(Peak1)
    E1_peak.append(Peak1)
    Freq_1.append(freq1)
  
    Peak2=max(Shift_Fourier_FinalE2)
    Shift_Fourier_FinalE2_list=list(Shift_Fourier_FinalE2)
    index2=Shift_Fourier_FinalE2_list.index(Peak2)
    value_before_index2=index2-1
    freq2=Shift_Frequency[value_before_index2:index2]
    print(freq2)
    print(Peak2)
    E2_peak.append(Peak2)
    Freq_2.append(freq2)
#Extra: Find Maximum




