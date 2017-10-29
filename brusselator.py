# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 01:07:59 2017

@author: Eddie
"""
import time
import numpy as np
from numba import jit
import scipy.linalg as slin
import numpy.linalg as nlin
#from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation


@jit
def RK4(xy):
    k1 = dt * f(xy)
    k2 = dt * f(xy + 0.5*k1)
    k3 = dt * f(xy + 0.5*k2)
    k4 = dt * f(xy + k3)
    return xy + (k1 + 2*k2 + 2*k3 + k4)/6

@jit
def animate(i):
# The plot shows the temperature evolving with time
    ax1.clear()
    plt.plot(x, R[i][0], color='red', label='time = '+str(round(t[i], 1))+'s')
    plt.plot(x, R[i][1], color='blue')
#    plt.plot(x, parB[i], color='black', linestyle = '--', linewidth = 1)
    plt.grid(True)
    plt.ylim([0, 8])
    plt.xlim([0, L])
#     plt.yticks(np.arange(np.min(R), np.max(R)+1, .5))
#    plt.xticks(np.arange(0, L+.1, 2.))
    plt.title('A = {}, B = {}, D = [{}, {}]'.format(A, B, D[0], D[1]))
    plt.legend(bbox_to_anchor=[1, 1])

@jit
def f(xy):
    return np.array([A + xy[1]*xy[0]*xy[0] - (B + 1)*xy[0], B*xy[0] - xy[1]*xy[0]*xy[0]])


def trid(Nx, ksi):
    a = np.ones(Nx+1)*(-ksi[:, np.newaxis])    # above main diag
    c = a.copy()                # under main diagonal
    a[:, 0], c[:, -1] = 0, 0
    a[:, 1] = -2*ksi
    c[:, -2] = -2*ksi
    b = np.ones(Nx+1)*(2*ksi[:, np.newaxis] + 1)    # main diag
    return np.array([np.vstack((a[i], b[i], c[i])) for i in [0, 1]])     # banded matrix for solve_banded()


if __name__ == '__main__':
    tic = time.clock()
    print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    L = 100                      # space
    Nx = 2000                   # num space points
    x = np.linspace(0, L, Nx+1) # mesh points in space
    D = np.array([1., 0.001])      # diff coeficient Dx Dy
    T = 40                      # final temperature
    dx = x[1]-x[0]              # space step
    dt = 0.02
    Nt = int(round(T/dt))       # num temp points
    t = np.linspace(0, T, Nt+1) # mesh points in time
    ksi = 0.5*D*dt/dx**2            # help var
    A, B = 0.5, 2.3
    initialFunc = 0.01*np.random.rand(2,Nx+1) + [[A], [B/A]]
    initialFunc[:,100:250] +=2
    initialFunc[:,700:900] +=2
#    initialFunc[0] = 6*np.sin(x)+1
#    initialFunc[1] = 6*np.cos(x+1)+1
#    initialFunc[0] = 6/(1+np.exp((x-2)/0.25))
#    initialFunc[1] = 7/(1+np.exp((x-3)/0.5))
    # B = 3*np.ones(Nx+1)
    # B[50:] = 1.9
    q = initialFunc.copy()
    R = []
    R.append(q.copy())
    a = np.ones(Nx+1)*(-ksi[:, np.newaxis])    # above main diag
    c = a.copy()                # under main diagonal
    a[:, 0], c[:, -1] = 0, 0
    a[:, 1] = -2*ksi
    c[:, -2] = -2*ksi
    b = np.ones(Nx+1)*(2*ksi[:, np.newaxis] + 1)    # main diag
    ab = np.array([np.vstack((a[i], b[i], c[i])) for i in [0, 1]])     # banded matrix for solve_banded()
    side = np.zeros((Nx+1,2))
#    parB = 5.1*np.ones((Nt+1,Nx+1))
#    parB[:,250:350] = 4
    for timeStep in range(Nt):
        # parB[timeStep,0+(timeStep):100+(timeStep)] = 6
#        B = parB[timeStep]
#        B = 1
        runge = RK4(q)
        side[0] = runge[:,0] + ksi*2*(q[:, 1] - q[:, 0])
        for i in range(1,Nx):
            side[i] = runge[:, i] + ksi*(q[:, i-1] - 2*q[:, i] + q[:, i+1])
        side[-1] = runge[:, -1] + ksi*2*(q[:, -2] - q[:, -1])
        q[0] = slin.solve_banded((1, 1), ab[0], side.T[0])
        q[1] = slin.solve_banded((1, 1), ab[1], side.T[1])
        R.append(q.copy())
        if timeStep % 100 == 0: print(timeStep, "/", Nt, time.strftime("%H:%M:%S", time.localtime()))
    print(timeStep, "/", Nt, time.strftime("%H:%M:%S", time.localtime()))
    R = np.array(R)
#==============================================================================
#   Print block
    plt.ioff()
    # plt.ion()
    plt.close('all')
    plt.style.use('fivethirtyeight')
    fig = plt.figure()
    fig.set_dpi(100)
    ax1 = fig.add_subplot(1, 1, 1)
    anim = animation.FuncAnimation(fig, animate, range(0, Nt+1,4), interval=10)
#    plt.plot(x,q[0], x,q[1])
#    plt.title('A = {}, B = {}, dx = {}, dt = {}'.format(A,B,dx,dt))
#    plt.savefig('{}.png'.format(time.strftime("%Y%m%d-%H%M%S")))
    FFMpegWriter = animation.writers['ffmpeg']
#    metadata = dict(title='1D Fizhugh-Nagumo', artist='Matplotlib', comment='Movie support!')
    writer = FFMpegWriter(fps=30, bitrate=1500)
    # anim.save('{}.mp4'.format(time.strftime("%Y%m%d-%H%M%S")), writer=writer)
    plt.show()
#==============================================================================
    print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    toc = time.clock()
    print("%5.3f" % (toc-tic))
