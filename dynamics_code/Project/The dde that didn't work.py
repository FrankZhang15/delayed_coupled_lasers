# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:07:22 2020

@author: Yande
"""
import numpy as np
import pylab as pl
import pydelay
from pydelay import dde23

# define the equations
eqns = {
    'x' : '0.25 * x(t-tau) / (1.0 + pow(x(t-tau),p)) -0.1*x'
    }

#define the parameters
params = {
    'tau': 15,
    'p'  : 10
    }

# Initialise the solver
dde = _dde23(eqns=eqns, params=params)

# set the simulation parameters
# (solve from t=0 to t=1000 and limit the maximum step size to 1.0)
dde.set_sim_params(tfinal=1000, dtmax=1.0)

# set the history of to the constant function 0.5 (using a python lambda function)
histfunc = {
    'x': lambda t: 0.5
    }
dde.hist_from_funcs(histfunc, 51)

# run the simulator
dde.run()

# Make a plot of x(t) vs x(t-tau):
# Sample the solution twice with a stepsize of dt=0.1:

# once in the interval [515, 1000]
sol1 = dde.sample(515, 1000, 0.1)
x1 = sol1['x']

# and once between [500, 1000-15]
sol2 = dde.sample(500, 1000-15, 0.1)
x2 = sol2['x']

pl.plot(x1, x2)
pl.xlabel('$x(t)$')
pl.ylabel('$x(t - 15)$')
pl.show()
