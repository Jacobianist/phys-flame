# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 16:37:24 2017

@author: Eddie
"""
import time
import numpy as np
from numba import jit
import scipy.linalg as slin
import numpy.linalg as nlin
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
# from matplotlib import rcParams
# this changes the dpi of figures saved from plt.savefig()
# rcParams['savefig.dpi'] = 300


# FitzHugh–Nagumo model function
# @jit
def f(xy):
    return np.array([500*(-xy[0]*(xy[0] - A)*(xy[0] - 1) - xy[1]), 
                        xy[0] - xy[1]])


# Runge Kutta 4th
# @jit
def RK4(xy):
    k1 = dt * f(xy)
    k2 = dt * f(xy + 0.5 * k1)
    k3 = dt * f(xy + 0.5 * k2)
    k4 = dt * f(xy + k3)
    return xy + (k1 + 2 * k2 + 2 * k3 + k4) / 6


# Tridiagonal matrix for Crank–Nicolson method to banded matrix for
# scipy.linalg.solve_banded()
def trid(Ny, ksi):
    a = np.ones(Ny+1)*(-ksi)    # above main diag
    c = a.copy()                # under main diagonal
    a[:, 0], c[:, -1] = 0, 0
    a[:, 1] = -2.*ksi.ravel()
    c[:, -2] = -2.*ksi.ravel()
    b = np.ones(Ny+1)*(2*ksi[:, np.newaxis] + 1)    # main diag
    return [np.vstack((a[i], b[i], c[i])) for i in [0, 1]]     # banded matrix for solve_banded()


# 1D Fisher-KPP model
def fn(L, Nx, x, dx, T, dt, Nt, w=1.):
    D = np.array([w, .0])           # diff coeficient Dx Dy
    ksi = 0.5*D*dt/dx**2            # help var
    initialFunc = np.zeros((2, Nx+1))
    initialFunc[0] = 1/(1+np.exp((x-1)/0.25))
    initialFunc[1] = 1/(1+np.exp((x-3)/0.5))
    q = initialFunc[:]
    R = []
    R.append(q.copy())
# =============================================================================
    a = np.ones(Nx+1)*(-ksi[:, np.newaxis])    # above main diag
    c = a.copy()                # under main diagonal
    a[:, 0], c[:, -1] = 0, 0
    a[:, 1] = -2.*ksi
    c[:, -2] = -2.*ksi
    b = np.ones(Nx+1)*(2*ksi[:, np.newaxis] + 1)    # main diag
    ab = np.array([np.vstack((a[i], b[i], c[i])) for i in [0, 1]])     # banded matrix for solve_banded()
# =============================================================================
    side = np.zeros((Nx+1, 2))

    @jit
    def f1(xy):
        return 7*np.array([xy[0]*(1-xy[0]), 5*xy[0]*(1-xy[0])-xy[1]])
    @jit
    def RK(xy):
        k1 = dt * f1(xy)
        k2 = dt * f1(xy + 0.5*k1)
        k3 = dt * f1(xy + 0.5*k2)
        k4 = dt * f1(xy + k3)
        return (xy + (k1 + 2*k2 + 2*k3 + k4)/6)
    for timeStep in range(2*Nt):
        runge = RK(q)
        side[0] = runge[:, 0] + ksi*2*(q[:, 1] - q[:, 0])
        for i in range(1, Nx):
            side[i] = runge[:, i] + ksi*(q[:, i-1] - 2*q[:, i] + q[:, i+1])
        side[-1] = runge[:, -1] + ksi*2*(q[:, -2] - q[:, -1])
        q = np.array(list(map(lambda x: slin.solve_banded((1, 1), ab[x], side.T[x]), [0, 1])))
        R.append(q.copy())
    R = np.array(R)
    return R

# one iteration of alternating direction implicit method
# with PERIODIC CONDITIONS
def solveFN(Nx, Ny, ksi, q, ab, A):
    runge = RK4(q)
    side0, side1 = np.zeros((2, Nx+1, Ny+1))
    for i in range(Nx+1):
        for j in range(1, Ny):
            side0[i,j], side1[i,j] = ksi*(q[:,i,j-1] - 2*q[:,i,j] + q[:,i,j+1]) + runge[:,i,j]
        side0[i,0], side1[i,0] = ksi*(q[:,i,-1] - 2*q[:,i,0] + q[:,i,1]) + runge[:,i,0]
        side0[i,-1], side1[i,-1] = ksi*(q[:,i,-2] - 2*q[:,i,-1] + q[:,i,0]) + runge[:,i,-1]
    q0 = np.array([nlin.solve(ab[0], side0[:,j]) for j in range(Ny+1)])
    q1 = np.array([nlin.solve(ab[1], side1[:,j]) for j in range(Ny+1)])
    q = np.array((q0.T, q1.T))

    for j in range(Ny+1):
        for i in range(1,Nx):
            side0[i,j], side1[i,j] = ksi*(q[:,i-1,j] - 2*q[:,i,j] + q[:,i+1,j]) + q[:,i,j]
        side0[0,j], side1[0,j] = ksi*(q[:,-1,j] - 2*q[:,0,j] + q[:,1,j]) + q[:,0,j]
        side0[-1,j], side1[-1,j] = ksi*(q[:,-2,j] - 2*q[:,-1,j] + q[:,0,j]) + q[:,-1,j]
    q0 = np.array([nlin.solve(ab[0], side0[i,:]) for i in range(Nx+1)])
    q1 = np.array([nlin.solve(ab[1], side1[i,:]) for i in range(Nx+1)])
    q = np.array((q0, q1))
    return q

# if __name__ == '__main__':
tic = time.clock()
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
L = 10                      # space
Nx = 100                    # space points
Ny = 100
x = np.linspace(0, L, Nx+1) # mesh points in space
y = np.linspace(0, L, Ny+1)
dx = x[1] - x[0]            # space step
T = 1                       # final time
dt = 0.005                  # time step
Nt = round(T/dt)            # time points
t = np.linspace(0, T, Nt+1) # mesh points in time
D = np.array([1, .0])       # diffusion coefficient Dx Dy
ksi = 0.5*D*dt/dx**2        # help var

# initial function
initialFunc = np.zeros((2, Nx+1, Ny+1))
initialFunc[0] = np.genfromtxt('1.out')[::2]
initialFunc[1] = np.genfromtxt('2.out')[::2]
# =============================================================================
# Initiation of transverse wave
rangeOf = fn(L, Nx, x, dx, T, dt, Nt, 0.001)
rangeOf = rangeOf[:, 1, :]
rangeOf = 0.6 - 0.5*rangeOf
# =============================================================================

q = initialFunc.copy()
ab = np.empty((2, Nx+1, Nx+1))
for i in range(2):
    ab[i] = np.diagflat(np.ones(Nx)*(-ksi[i]),-1) +\
            np.diagflat(np.ones(Nx+1)*(2*ksi[i] + 1)) +\
            np.diagflat(np.ones(Nx)*(-ksi[i]), 1)
    ab[i][0,-1] = -ksi[i]
    ab[i][-1,0] = -ksi[i]

parA = 10
Qu, Qv = [], [] # all resolve

for iteration in range(Nt):
    A = np.meshgrid(rangeOf[parA], rangeOf[parA])[1]
    q = solveFN(Nx, Ny, ksi, q, ab, A)
    Qu.append(q[0])
    Qv.append(q[1])
    parA += 1
    if iteration % 100 == 0: print(iteration,"/",Nt,"||",time.strftime("%H:%M:%S", time.localtime()))
Qu, Qv = np.asarray(Qu), np.asarray(Qv)

# =============================================================================
# Plot block
# plt.ioff()
# plt.ion()
plt.close('all')
plt.style.use('fivethirtyeight')
fig = plt.figure()
im = plt.imshow(Qu[0], animated=True, vmin=np.min(Qu), vmax=np.max(Qu), extent=(0,L,L,0))
ax = plt.gca()
ax.grid(False)
plt.colorbar()
def animate(z):
    im.set_array(Qu[z])
    plt.suptitle('time: {:03f}s'.format(t[z]))
    plt.tight_layout()
    return im,
anim = animation.FuncAnimation(fig, animate, range(0, Nt), interval=10)
# =============================================================================
# Animation save block
# FFMpegWriter = animation.writers['ffmpeg']
# metadata = dict(title='heat dynamic', artist='Matplotlib')
# writer = FFMpegWriter(fps=60, bitrate=1500)
# print(time.strftime("%H:%M:%S", time.localtime()), 'animation saving...')
# anim.save('FN_KPP_PBC_{}.mp4'.format(time.strftime("%Y%m%d-%H%M%S")), writer=writer)
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
toc = time.clock()
print("%5.3f" % (toc-tic))
plt.show()
