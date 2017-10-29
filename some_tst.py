# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 09:52:44 2017

@author: Eddie
"""

import time
import numpy as np
from numba import jit

from scipy.integrate import odeint

@jit
def f(xy):
    return np.array([(-xy[0]*(xy[0]-0.1)*(xy[0]-1) - xy[1])*500, xy[0] - xy[1]])

@jit
def RK4(xy):
    k1 = dt * f(xy)
    k2 = dt * f(xy + 0.5*k1)
    k3 = dt * f(xy + 0.5*k2)
    k4 = dt * f(xy + k3)
    return (xy + (k1 + 2*k2 + 2*k3 + k4)/6)#[:, np.newaxis]

if __name__ == '__main__':
#    tic = time.clock()
#    print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    dt = 0.005
    L = 1                     # space
    Nx = 100                   # num space points
    Ny = 100
    x = np.linspace(0, L, Nx+1)   # mesh points in space
    y = np.linspace(0, L, Ny+1)
    uv = np.random.random((2, Nx+1, Ny+1))
#    f1 = f(uv)
#    df = RK4(uv)
    qw = np.empty((2,Nx+1,Ny+1))
    for i in range(Nx+1):
        for j in range(Ny+1):
            qw[:,i,j] = RK4(uv[:,i,j])