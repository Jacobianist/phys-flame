import time
import numpy as np
from numba import jit
import scipy.linalg as slin
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
import seaborn as sns

rcParams['savefig.dpi'] = 300

def RK4(xy):
    f = lambda xy: np.array([(-xy[0]*(xy[0] - 0.1)*(xy[0]-1) - xy[1])*500, xy[0] - xy[1]])
    k1 = dt * f(xy)
    k2 = dt * f(xy + 0.5*k1)
    k3 = dt * f(xy + 0.5*k2)
    k4 = dt * f(xy + k3)
    return xy + (k1 + 2*k2 + 2*k3 + k4)/6


dt = 0.005
L = 10                      # space
Nx = 100                   # num space points
x = np.linspace(0, L, Nx+1) # mesh points in space
initialFunc = np.zeros((2, Nx+1))+0.1
q = initialFunc[:]
