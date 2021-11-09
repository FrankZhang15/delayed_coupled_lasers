# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:31:29 2020

@author: User
"""

import matplotlib.pyplot as plt
import scipy as sp
from pylab import linspace, array
from ddeint import ddeint
from scipy.optimize import fsolve
import pylab
import numpy as np

#%%
'''
Testing section

kappa=0.2
Cp=0.1*sp.pi
alpha=2.6
tao=0.2

x = npy.linspace(-2,2,500)
def y1(x):
    y1=  0.2 * ((1+2.6**2)**0.5) * npy.sin(0.1*sp.pi+0.5*x+npy.arctan(2.6))
    return y1
def y2(x):
    y2= -0.2 * ((1+2.6**2)**0.5) * npy.sin(0.1*sp.pi+0.5*x+npy.arctan(2.6))
    return y2
def y3(x):
    y3 =x
    return y3

def find_Intersection(fun1,fun2,x0):
    return fsolve(lambda x : fun1(x) - fun2(x),x0)
result = find_Intersection(y1,y3,0.0)


print(result)
'''

#%%
'''Official code
   variables defined here are just for checking purpose, they will be defined
   again in the code.'''
'''alpha=2.6
   tao=0.2
   kappa=0.2
   tao_p=7.7*(10**(-12))
   tao_norm=25974025974.025974
'''
'''make the loop for Cp, set a=2.6'''
'''w= +/- k(1+a^2)^0.5 x sin(Cp + tao*w + arctan(a))'''
'''I seperate the equation into 2 parts, left hand side(y3) 
   and right hand side(y1,y2). The '''
#%%
'''investigating Cp and w(freq)'''   

'''w=k(1+a^2)^0.5 x sin(Cp + tao*w + arctan(a))'''
   
tao_p=7.7*(10**(-12))
#devided by pi to fit into X-axis
   
Cps=sp.linspace(0,sp.pi,500)
ks= sp.linspace(0.1,0.5,5)
c=[0,1,2,3,4]
colour=['red','orange','yellow','green','blue']

# Define the boudary of plotting below (to exclude the 2 colour region):
one_colour_low =[0.18,0.22,0.35,0.50,0.53]
one_colour_high=[0.52]
    
    

for k in ks:
    # Add a interger form of k so that we could use it as an index
    k_interger=int((k-0.1)*10)
    
    Cp_wanted_low =[]# Set up this list to exclude the Cp for 2 colour region
    Cp_wanted_high=[]# Set up this list to exclude the Cp for 2 colour region
    
# All the plotted datas, save them so that someone else could plot the graph as well
    result_w_in       =[]#high
    result_w_anti     =[]#low
    normalised_Cp_in  =[]# high
    normalised_Cp_anti=[]# low
    normalised_Cp_all =[]# summation of low and high Cp, executed near the end of code
    result_all        =[]# Same as above
    alpha=2.6
    
    
    Cp_wanted_low =Cps[ Cps < (one_colour_low[k_interger])*sp.pi ] 
    Cp_wanted_high=Cps[ Cps>0.52*sp.pi ]
               
    for Cp in Cp_wanted_low: 
        def y1(w):
            y1=  k * ((1+alpha**2)**0.5) * np.sin(Cp+0.2*w+np.arctan(2.6))
            return y1
        def y2(w):
            y2= -k * ((1+alpha**2)**0.5) * np.sin(Cp+0.2*w+np.arctan(2.6))
            return y2
        def y3(w):
            y3 =w
            return y3
    
        def find_Intersection_inphase  (fun1,fun3,w0):
            return fsolve(lambda w : fun1(w) - fun3(w),w0)
    
        def find_Intersection_antiphase(fun2,fun3,w0):
            return fsolve(lambda w:  fun2(w) - fun3(w),w0)

        w = np.linspace(-10,10,500)
        result_anti= find_Intersection_antiphase(y2,y3,0.0) # Un-normalised result, needed to be normalised
    
        result_anti_norm=(result_anti)/tao_p
        result_w_anti.append(result_anti_norm)
    
        normalisation=Cp/sp.pi
        normalised_Cp_anti.append(normalisation)
    
   
    for Cp2 in Cp_wanted_high: 
        def y1(w):
            y1=  k * ((1+alpha**2)**0.5) * np.sin(Cp2+0.2*w+np.arctan(2.6))
            return y1
        def y2(w):
            y2= -k * ((1+alpha**2)**0.5) * np.sin(Cp2+0.2*w+np.arctan(2.6))
            return y2
        def y3(w):
            y3 =w
            return y3

        def find_Intersection_inphase  (fun1,fun3,w0):
            return fsolve(lambda w : fun1(w) - fun3(w),w0)
    
        def find_Intersection_antiphase(fun2,fun3,w0):
            return fsolve(lambda w:  fun2(w) - fun3(w),w0)

        w = np.linspace(-10,10,500)
        result_in  = find_Intersection_inphase  (y1,y3,0.0) # Un-normalised result, needed to be normalised
  
        result_in_norm=result_in/tao_p
        result_w_in.append(result_in_norm)
    
        normalisation2=Cp2/sp.pi
        normalised_Cp_in.append(normalisation2)
            
   
    normalised_Cp_all=normalised_Cp_anti+normalised_Cp_in
    result_all=result_w_anti+result_w_in
    #sp.savetxt('Cp_k_'+str(k)+'.csv',normalised_Cp_all,delimiter=',' )
    #sp.savetxt('fq_k_'+str(k)+'.csv',result_all       ,delimiter=',' )
    
    result_all_array=(np.array(result_all))  *1e-9
    plt.scatter(normalised_Cp_all, result_all_array, marker='o', s=0.05, 
                c=colour[k_interger])
    plt.grid()
       
plt.xlabel('Cp/pi')
plt.ylabel('frequency in GHz')
plt.show()
#%%
'''investigating alpha and w(freq)'''
Cp=0.1*sp.pi # For this Cp value, the region will be at the antiphase quadron, so we take the antiphase solution
Alphas=sp.linspace(1,8,800)
ks= sp.linspace(0.1,0.5,5)
tao_p=7.7*(10**(-12))
for k in ks:
    # Add a interger form of k so that we could use it as an index
    
    k_interger=int((k-0.1)*10)
    result_a=[]

    
    for alpha in Alphas: 
        def y1(w):
            y1=  k * ((1+alpha**2)**0.5) * np.sin(Cp+0.2*w+np.arctan(2.6))
            return y1
        def y2(w):
            y2= -k * ((1+alpha**2)**0.5) * np.sin(Cp+0.2*w+np.arctan(2.6))
            return y2
        def y3(w):
            y3 =w
            return y3
    
        def find_Intersection_inphase  (fun1,fun3,w0):
            return fsolve(lambda w : fun1(w) - fun3(w),w0)
    
        def find_Intersection_antiphase(fun2,fun3,w0):
            return fsolve(lambda w:  fun2(w) - fun3(w),w0)

        w = np.linspace(-3,3,800)
        result= find_Intersection_antiphase(y2,y3,0.0) # Un-normalised result, needed to be normalised
    
        result_norm=(result)/tao_p*1e-9
        result_a.append(result_norm)
    
    plt.scatter(Alphas, result_a,marker='x',s=0.05)
    plt.legend(str(k))
    plt.grid()
plt.xlabel('alpha')
plt.ylabel('freqency in GHz')
plt.show()
        






#%%
'''
All the problems you have faced in this code:
1) Colouring problems caused by position of the list ( List outside the loop causing overplotting)
2) 
'''


    

    
    
    
    
    
    
    
    
    
    
    
    

