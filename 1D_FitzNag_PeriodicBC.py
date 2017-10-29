# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 01:07:59 2017

@author: Eddie
"""
import time
import sys
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
    plt.grid(True)
    plt.ylim([np.min(R), np.max(R)])
    plt.xlim([0, L])
    # plt.yticks(np.arange(np.min(R), np.max(R)+1, .5))
    plt.xticks(np.arange(0, L+.1, 1.))
#    plt.title('tau = {:.1e}, dx = {:.1e}, dt = {:.1e}'.format(tau, dx, dt))
    plt.legend(bbox_to_anchor=[1, 1.1])

@jit
def f(xy):
    return np.array([(-xy[0]*(xy[0]-A)*(xy[0]-1) - xy[1])*250, xy[0] - xy[1]])

if __name__ == '__main__':
    tic = time.clock()

    print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    L = 10                      # space
    Nx = 200                   # num space points
    x = np.linspace(0, L, Nx+1) # mesh points in space
    D = np.array([0.5, .0])      # diff coeficient Dx Dy
    T = 3                       # final temperature
    dx = x[1]-x[0]              # space step
#    dt = 0.9*dx*dx/2           # temp step /D[1]
    dt = 0.005
    Nt = int(round(T/dt))       # num temp points
    t = np.linspace(0, T, Nt+1) # mesh points in time
    ksi = 0.5*D*dt/dx**2            # help var

# initial functions
    initialFunc = np.zeros((2, Nx+1))
    # initialFunc[0][:5] += 0.001
#    initialFunc[0] = np.genfromtxt('test.out')[0::10]
    initialFunc[0] = np.genfromtxt('1.out')
    initialFunc[1] = np.genfromtxt('2.out')
#    initialFunc[0] = 1/(1+np.exp((x*5-1)/0.25))
    # initialFunc = np.random.rand(2,Nx+1)+0.1

    q = initialFunc[:]
    R = []
    R.append(q.copy())
    ab = np.empty((2, Nx+1, Nx+1))
    for i in range(2):
        ab[i] = np.diagflat(np.ones(Nx)*(-ksi[i]),-1) +\
                np.diagflat(np.ones(Nx)*(-ksi[i]), 1) +\
                np.diagflat(np.ones(Nx+1)*(2*ksi[i] + 1))
        ab[i][0,-1] = -ksi[i]
        ab[i][-1,0] = -ksi[i]
    side = np.zeros((Nx+1,2))
    A = 0.1
    for timeStep in range(Nt):
# right-hand side vector
        runge = RK4(q)
        side[0] = runge[:,0] + ksi*(q[:,-1] - 2*q[:,0] + q[:,1])
        for i in range(1,Nx):
            side[i] = runge[:, i] + ksi*(q[:, i-1] - 2*q[:, i] + q[:, i+1])
        side[-1] = runge[:, -1] + ksi*(q[:,-2] - 2*q[:,-1] + q[:,0])
        q[0] = nlin.solve(ab[0], side.T[0])
        q[1] = nlin.solve(ab[1], side.T[1])
        R.append(q.copy())
        if timeStep % 100 == 0: print(timeStep,"/",Nt,time.strftime("%H:%M:%S", time.localtime()))
        sys.stdout.flush()
    R = np.array(R)
#==============================================================================
#   Print block
#    plt.ioff()
#    plt.ion()
#    plt.close('all')
    fig = plt.figure()
    plt.style.use('fivethirtyeight')
    # fig.set_dpi(100)
    ax1 = fig.add_subplot(1, 1, 1)
    anim = animation.FuncAnimation(fig, animate, range(0, Nt+1), interval=10)
    plt.show()


#    FFMpegWriter = animation.writers['ffmpeg']
#    metadata = dict(title='1D Fizhugh-Nagumo', artist='Matplotlib', comment='Movie support!')
#    writer = FFMpegWriter(fps=30, bitrate=1500)#, metadata=metadata)
#    anim.save('{}.mp4'.format(time.strftime("%Y%m%d-%H%M%S")), writer=writer)
#==============================================================================
    print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    toc = time.clock()
    print("%5.3f" % (toc-tic))
