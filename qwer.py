# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 23:28:05 2017

@author: Eddie
"""
import time
import numpy as np
#from numba import jit, autojit, vectorize, float64, float32
import scipy.linalg as slin
#from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
# this changes the dpi of figures saved from plt.savefig()
rcParams['savefig.dpi'] = 300


def f(xy):
    return np.array([(-xy[0]*(xy[0]-A)*(xy[0]-1) - xy[1])*500, xy[0] - xy[1]])


def RK4(xy):
    k1 = dt * f(xy)
    k2 = dt * f(xy + 0.5*k1)
    k3 = dt * f(xy + 0.5*k2)
    k4 = dt * f(xy + k3)
    return (xy + (k1 + 2*k2 + 2*k3 + k4)/6)[:, np.newaxis]


def hmm(i, j, it):
    d = round(it/10)
    if 5 + d < i < 10 + d:
        if j < 5:
            return -0.1
        else:
            return 0.1
    else:
        return 0.4


def trid(Ny, ksi):
    a = np.ones(Ny+1)*(-ksi)    # above main diag
    c = a.copy()              # under main diagonal
    a[:, 0], c[:, -1] = 0, 0
    a[:, 1] = -2*ksi.ravel()
    c[:, -2] = -2*ksi.ravel()
    b = np.ones(Ny+1)*(2*ksi[:, np.newaxis] + 1)    # main diag
    return [np.vstack((a[i], b[i], c[i])) for i in [0, 1]]     # banded matrix for solve_banded()


if __name__ == '__main__':
    tic = time.clock()
#    plt.ioff()
    print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))

    L = 10                     # space
    Nx = 100                   # num space points
    Ny = 100
    x = np.linspace(0, L, Nx+1)   # mesh points in space
    y = np.linspace(0, L, Ny+1)
    D = np.array([1, .0])     # diff coeficient Dx Dy
    T = .5                    # final temperature
    dx = x[1] - x[0]           # space step
    dt = 0.005
    Nt = round(T/dt)      # num temp points
    t = np.linspace(0, T, Nt+1)  # mesh points in time
    ksi = (0.5*D*dt/dx**2)[:, np.newaxis]       # help var

    xx, yy = np.meshgrid(x, y)
    initialFunc = np.zeros((2, Nx+1, Ny+1))
    initialFunc[0, :10, :51] = R[100, 0, :][0::20]
    initialFunc[1, :10, :51] = R[100, 1, :][0::20]
    q = initialFunc.copy()
    ab = trid(Nx, ksi)
#    # additional vars for tridiagonal solver
#    a = np.ones(Ny+1)*(-ksi[:,np.newaxis])    # above main diag
#    c = a.copy()              # under main diagonal
#    a[:, 0], c[:, -1] = 0, 0
#    a[:, 1] = -2*ksi
#    c[:, -2] = -2*ksi
#    ksi = ksi[:,np.newaxis]
#    b = np.ones(Ny+1)*(2*ksi + 1)    # main diag
#    ab = [np.vstack((a[i], b[i], c[i])) for i in [0, 1]]     # banded matrix for solve_banded()
    Qu, Qv = [], []
    for iteration in range(Nt):
        side0, side1 = np.zeros((2, Nx+1, Ny+1))
        for i in range(Nx+1):
            for j in range(1,Ny):
                A = hmm(i,j,iteration)
                side0[i,j], side1[i,j] = ksi*(q[:,i,j-1] - 2*q[:,i,j] + q[:,i,j+1])[:,np.newaxis] + RK4(q[:,i,j])
            A = hmm(i,0,iteration)
            side0[i,0], side1[i,0] = ksi*2*(q[:,i,1] - q[:,i,0])[:,np.newaxis] + RK4(q[:,i,0])
            A = hmm(i,-1,iteration)
            side0[i,-1], side1[i,-1] = ksi*2*(q[:,i,-2] - q[:,i,-1])[:,np.newaxis] + RK4(q[:,i,-1])
        q0 = np.array([slin.solve_banded((1, 1), ab[0], side0[:,j]) for j in range(Ny+1)])
        q1 = np.array([slin.solve_banded((1, 1), ab[1], side1[:,j]) for j in range(Ny+1)])
        q = np.array((q0.T, q1.T))
        for i in range(1, Nx):
            for j in range(Ny+1):
                A = hmm(i,j,iteration)
                side0[i,j], side1[i,j] = ksi*(q[:,i-1,j] - 2*q[:,i,j] + q[:,i+1,j])[:,np.newaxis] + q[:,i,j][:,np.newaxis]
                A = hmm(0,j,iteration)
                side0[0,j], side1[0,j] = ksi*2*(q[:,1,j] - q[:,0,j])[:,np.newaxis] + q[:,0,j][:,np.newaxis]
                A = hmm(-1,j,iteration)
                side0[-1,j], side1[-1,j] = ksi*2*(q[:,-2,j] - q[:,-1,j])[:,np.newaxis] + q[:,-1,j][:,np.newaxis]
        q0 = np.array([slin.solve_banded((1, 1), ab[0], side0[i,:]) for i in range(Nx+1)])
        q1 = np.array([slin.solve_banded((1, 1), ab[1], side1[i,:]) for i in range(Nx+1)])
        q = np.array((q0, q1))
        Qu.append(q[0])
        Qv.append(q[1])
        if iteration % 10 == 0: print(iteration,"/",Nt,"||",time.strftime("%H:%M:%S", time.localtime()))
    Qu, Qv = np.asarray(Qu), np.asarray(Qv)

#==============================================================================
# Plotting block
    plt.close('all')
    fig = plt.figure()
    im = plt.imshow(Qu[0], animated=True)
    plt.colorbar()
    def animate(z):
        im.set_array(Qu[z])
        return im,
    anim = animation.FuncAnimation(fig, animate, range(0, Nt), interval=10)
#    FFMpegWriter = animation.writers['ffmpeg']
#    metadata = dict(title='heat dynamic', artist='Matplotlib')
#    writer = FFMpegWriter(fps=60, bitrate=1000, metadata=metadata)
#    anim.save('big_setup.mp4', writer=writer)
#==============================================================================
#    plt.show()
    print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    toc = time.clock()
    print("%5.3f" % (toc-tic))
