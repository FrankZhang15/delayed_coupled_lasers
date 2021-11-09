# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:06:24 2020

@author: Yande
"""

import matplotlib.pyplot as plt
import numpy as npy
import scipy as sp
from pylab import linspace, array
from ddeint import ddeint

#%%
#Define Variables
E1_peak_neg=[]
Freq1_neg=[]
E2_peak_neg=[]
Freq2_neg=[]
E1_peak_pos=[]
Freq1_pos=[]
E2_peak_pos=[]
Freq2_pos=[]

T=392
P=0.23
tao_p=7.7E-12
alpha=2.6


Er10=0.4
Ei10=0
Er20=-0.4
Ei20=0
N10=0.05
N20=-0.05

tao=0.2
kappa=0.2


tt_minimum=0
tt_maximum=12000
tt_space=60000

good_t=[70,84]

params = {
   'axes.labelsize': 12,
   'font.size': 12,
   'legend.fontsize': 12,
   'xtick.labelsize': 16,
   'ytick.labelsize': 16,
   'figure.figsize': [8, 6]
   } 
#%%
'''Changing Alpha'''
#While loop generates an array of alpha:
Cp=0.0*sp.pi
limit=1*sp.pi
Cps=[]
while Cp < limit:
    Cps.append(Cp)
    Cp += 0.03
print(Cps)
#%%
#creat FOR loop for codes:

for Cp in Cps:
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





    t_axis=(sp.array(tt)*tao_p) *1E9






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




    

    Peak1_neg=max(Shift_Fourier_FinalE1[:4999])
    Shift_Fourier_FinalE1_list=list(Shift_Fourier_FinalE1)
    index1_neg=Shift_Fourier_FinalE1_list.index(Peak1_neg)
    value_before_index1_neg=index1_neg-1
    freq1_neg=Shift_Frequency[value_before_index1_neg:index1_neg]
    print(freq1_neg)
    print(Peak1_neg)
    E1_peak_neg.append(Peak1_neg)
    Freq1_neg.append(freq1_neg)
    
    Peak1_pos=max(Shift_Fourier_FinalE1[4999:])
    Shift_Fourier_FinalE1_list=list(Shift_Fourier_FinalE1)
    index1_pos=Shift_Fourier_FinalE1_list.index(Peak1_pos)
    value_before_index1_pos=index1_pos-1
    freq1_pos=Shift_Frequency[value_before_index1_pos:index1_pos]
    print(freq1_pos)
    print(Peak1_pos)
    E1_peak_pos.append(Peak1_pos)
    Freq1_pos.append(freq1_pos)
  
    Peak2_pos=max(Shift_Fourier_FinalE2[4999:])
    Shift_Fourier_FinalE2_list=list(Shift_Fourier_FinalE2)
    index2_pos=Shift_Fourier_FinalE2_list.index(Peak2_pos)
    value_before_index2_pos=index2_pos-1
    freq2_pos=Shift_Frequency[value_before_index2_pos:index2_pos]
    print(freq2_pos)
    print(Peak2_pos)
    E2_peak_pos.append(Peak2_pos)
    Freq2_pos.append(freq2_pos)
    
    Peak2_neg=max(Shift_Fourier_FinalE2[:4999])
    Shift_Fourier_FinalE2_list=list(Shift_Fourier_FinalE2)
    index2_neg=Shift_Fourier_FinalE2_list.index(Peak2_neg)
    value_before_index2_neg=index2_neg-1
    freq2_neg=Shift_Frequency[value_before_index2_neg:index2_neg]
    print(freq2_neg)
    print(Peak2_neg)
    E2_peak_neg.append(Peak2_neg)
    Freq2_neg.append(freq2_neg)
    

    plt.rcParams.update(params)
    plt.plot(t_axis, Intensity1, lw=1, color='orange', label="delay = %.01f" % tao)
    plt.plot(t_axis, Intensity2, lw=1, color='blue', label="delay = %.01f" % tao)
    plt.xlim(good_t)
    plt.ylabel("Intensity")
    plt.xlabel('Time (ns)')
    plt.title("Cp=%.03f_Intensity\nTau= %.03f, k=%.3f, a=%.03f, T= %.01f, P= %.03f, Cp= %.03f\nGraph" % (Cp,tao,kappa,alpha,T,P,Cp))
    plt.legend(['Intensity1','Intensity2'],loc=1)
    plt.savefig(f"Cp=%.03f_Intensity_Specific.png" % (Cp),bbox_inches = 'tight')
    plt.show()
    
    
    plt.rcParams.update(params)
    plt.plot(t_axis, Intensity1, lw=1, color='orange', label="delay = %.01f" % tao)
    plt.plot(t_axis, Intensity2, lw=1, color='blue', label="delay = %.01f" % tao)
    plt.ylabel("Intensity")
    plt.xlabel('Time (ns)')
    plt.title("Cp=%.03f_Intensity\nTau= %.03f, k=%.3f, a=%.03f, T= %.01f, P= %.03f, Cp= %.03f\nGraph" % (Cp,tao,kappa,alpha,T,P,Cp))
    plt.legend(['Intensity1','Intensity2'],loc=1)
    plt.savefig(f"Cp=%.03f_Intensity_All.png"% (Cp),bbox_inches = 'tight')
    plt.show()
    
    plt.rcParams.update(params)
    plt.plot(t_axis, N1, lw=1, color='orange', label="delay = %.01f" % tao)
    plt.plot(t_axis, N2, lw=1, color='blue', label="delay = %.01f" % tao)
    plt.xlabel("Time (ns)")
    plt.ylabel("Inversion")
    plt.legend(['N1','N2'],loc=1)
    plt.title("a=Cp=%.03f_Inversion\nTau= %.03f, k=%.3f, a=%.03f, T= %.01f, P= %.03f, Cp= %.03f\nGraph" % (Cp,tao,kappa,alpha,T,P,Cp))
    plt.savefig(f"Cp=%.03f_Inversion_All.png"% (Cp),bbox_inches = 'tight')
    plt.show()
    
    
    plt.rcParams.update(params)
    plt.plot(t_axis, N1, lw=1, color='orange', label="delay = %.01f" % tao)
    plt.plot(t_axis, N2, lw=1, color='blue', label="delay = %.01f" % tao)
    plt.xlim(good_t)
    plt.xlabel("Time (ns)")
    plt.ylabel("Inversion")
    plt.legend(['N1','N2'],loc=1)
    plt.title("Cp=%.03f_Inversion\nTau= %.03f, k=%.3f, a=%.03f, T= %.01f, P= %.03f, Cp= %.03f\nGraph" % (Cp,tao,kappa,alpha,T,P,Cp))
    plt.savefig(f"Cp=%.03f_Inversion_Specific.png"% (Cp),bbox_inches = 'tight')
    plt.show()
    
    plt.rcParams.update(params)
    plt.plot(Shift_Frequency, Shift_Fourier_FinalE1, lw=1, color='orange', label="delay = %.01f" % tao)
    plt.xlim([-2E11,2E11])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Fourier_of_E1")
    plt.legend(['Fourier_E1'],loc=1)
    plt.title("Cp=%.03f_Frequency1\nTau= %.03f, k=%.3f, a=%.03f, T= %.01f, P= %.03f, Cp= %.03f\nGraph" % (Cp,tao,kappa,alpha,T,P,Cp))
    plt.savefig(f"Cp=%.03f_Frequency1.png"% (Cp),bbox_inches = 'tight')
    plt.show()
    
    plt.rcParams.update(params)
    plt.plot(Shift_Frequency, Shift_Fourier_FinalE2, lw=1, color='blue', label="delay = %.01f" % tao)
    plt.xlim([-2E11,2E11])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Fourier_of_E2")
    plt.legend(['Fourier_E2'],loc=1)
    plt.title("Cp=%.03f_Frequency2\nTau= %.03f, k=%.3f, a=%.03f, T= %.01f, P= %.03f, Cp= %.03f\nGraph" % (Cp,tao,kappa,alpha,T,P,Cp))
    plt.savefig(f"Cp=%.03f_Frequency2.png"% (Cp),bbox_inches = 'tight')
    plt.show()

#%%
'''Checking the numbers and plotting some graphs, run below after loops'''
Cps=npy.array(Cps)
Cps_pi=Cps/sp.pi
Freq_fit=[]
Freq_fit_else=[]
for i in range(0,24):
    Freq_fit.append(Freq1_neg[i])
for i in range(24,54):
    Freq_fit.append(Freq1_pos[i])
for i in range(54,65):
    Freq_fit.append(Freq1_pos[i])
for i in range(65,105):
    Freq_fit.append(Freq1_neg[i])

for i in range(24,54):
    Freq_fit_else.append(Freq1_neg[i])

    
Freq_fit=npy.array(Freq_fit)
Freq_fit_else=npy.array(Freq_fit_else)
    

Freq_fit2=[]
Freq_fit_else2=[]
for i in range(0,24):
    Freq_fit2.append(Freq2_neg[i])
for i in range(24,54):
    Freq_fit2.append(Freq2_pos[i])
for i in range(54,65):
    Freq_fit2.append(Freq2_pos[i])
for i in range(65,105):
    Freq_fit2.append(Freq2_neg[i])

for i in range(24,54):
    Freq_fit_else2.append(Freq2_neg[i])

    
Freq_fit2=npy.array(Freq_fit2)
Freq_fit_else2=npy.array(Freq_fit_else2)
    

Int1=[]
Int2=[]
Int1_else=[]
Int2_else=[]
for i in range(0,24):
    Int1.append(E1_peak_neg[i])
for i in range(24,54):
    Int1.append(E1_peak_pos[i])
for i in range(54,65):
    Int1.append(E1_peak_pos[i])
for i in range(65,105):
    Int1.append(E1_peak_neg[i])
for i in range(24,54):
    Int1_else.append(E1_peak_neg[i])
Int1=npy.array(Int1)
Int1_else=npy.array(Int1_else)

for i in range(0,24):
    Int2.append(E2_peak_neg[i])
for i in range(24,54):
    Int2.append(E2_peak_pos[i])
for i in range(54,65):
    Int2.append(E2_peak_pos[i])
for i in range(65,105):
    Int2.append(E2_peak_neg[i])
for i in range(24,54):
    Int2_else.append(E2_peak_neg[i])
Int2=npy.array(Int2)
Int2_else=npy.array(Int2_else)

#%%
Theo_Cp=sp.loadtxt('Cp/Cp_k_0.2.csv')
Theo_Fq=sp.loadtxt('W/fq_k_0.2.csv')
#%%
print(E1_peak_neg)
print(Freq1_neg)
print(E2_peak_neg)
print(Freq2_neg) 
print(E1_peak_pos)
print(Freq1_pos)
print(E2_peak_pos)
print(Freq2_pos) 

#%%
'''plottin the graph!'''

plt.errorbar(Cps_pi,Int1,fmt=".", mew=2, ms=6, capsize=3,color="blue")
plt.errorbar(Cps_pi[24:54],Int1_else,fmt=".", mew=2, ms=6, capsize=3,color="blue")
plt.plot()
#plt.xlim([1,6])
plt.xlabel("Cp/pi")
plt.ylabel("E1_peak")

plt.title("E1peak\nTau= %.03f, k=%.3f,  T= %.01f, P= %.03f\nGraph" % (tao,kappa,T,P))
plt.savefig(f"E1_peak_Cp.png",bbox_inches = 'tight')

#%%
plt.errorbar(Cps_pi,Int2,fmt=".", mew=2, ms=6, capsize=3,color="blue")
plt.errorbar(Cps_pi[24:54],Int2_else,fmt=".", mew=2, ms=6, capsize=3,color="blue")
#plt.xlim([2.5,3.3])
plt.xlabel("Cp/pi")
plt.ylabel("E2_peak")
plt.title("E2peak\nTau= %.03f, k=%.3f,  T= %.01f, P= %.03f\nGraph" % (tao,kappa,T,P))
plt.savefig(f"E2_peak_Cp.png",bbox_inches = 'tight')

#%%
plt.errorbar(Cps_pi,Freq_fit/1e9,fmt=".", mew=2, ms=6, capsize=3,color="blue")
plt.errorbar(Cps_pi[24:54],Freq_fit_else/1e9,fmt=".", mew=2, ms=6, capsize=3,color="blue")
plt.scatter(Theo_Cp,Theo_Fq/1e9, color="red")

#plt.xlim([1,3.32])
plt.ylim([-120,40])
plt.xlabel("Cp/pi")
plt.ylabel("Frequency for E1 (GHz)")
plt.legend(["E1_Analytical","E1_Numerical"])
plt.title("frequencyE1\nTau= %.03f, k=%.3f,  T= %.01f, P= %.03f\nGraph" % (tao,kappa,T,P))
plt.savefig(f"FrequencyE1_Cp.png",bbox_inches = 'tight')
#%%
plt.errorbar(Cps_pi,Freq_fit2/1e9,fmt=".", mew=2, ms=6, capsize=3,color="blue")
plt.errorbar(Cps_pi[24:54],Freq_fit_else2/1e9,fmt=".", mew=2, ms=6, capsize=3,color="blue")
plt.scatter(Theo_Cp,Theo_Fq/1e9, color="red")
#plt.xlim([1,3.32])
plt.ylim([-120,40])
plt.xlabel("Cp/pi")
plt.ylabel("Frequency for E2 (GHz)")
plt.legend(["E2_Analytical","E2_Numerical"])
plt.title("frequencyE2\nTau= %.03f, k=%.3f,  T= %.01f, P= %.03f\nGraph" % (tao,kappa,T,P))
plt.savefig(f"FrequencyE2_Cp.png",bbox_inches = 'tight')

#%%
npy.savetxt("Freq_fit.csv", Freq_fit, delimiter=",")
npy.savetxt("Freq_fit_else.csv", Freq_fit_else, delimiter=",")
npy.savetxt("Cps.csv", Cps_pi, delimiter=",")
npy.savetxt("Freq_fit2.csv", Freq_fit2, delimiter=",")
npy.savetxt("Freq_fit_else2.csv", Freq_fit_else2, delimiter=",")
npy.savetxt("Int1.csv", Int1, delimiter=",")
npy.savetxt("Int2.csv", Int2, delimiter=",")
npy.savetxt("Int1_else.csv", Int1_else, delimiter=",")
npy.savetxt("Int2_else.csv", Int2_else, delimiter=",")
npy.savetxt("Int2.csv", Int2, delimiter=",")