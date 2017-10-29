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
#from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

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
    plt.plot(x, R[i][0], color='red', label='u')#+str(round(t[i], 2))+'s')
    plt.plot(x, R[i][1], color='blue', label='v')
#    plt.plot(x, ff, '--', color='black')
    plt.grid(True)
    plt.ylim([np.min(R), np.max(R)])
    plt.xlim([0, L])
    plt.xlabel('time: {:03f}s'.format(t[i]))
    # plt.yticks(np.arange(np.min(R), np.max(R)+1, .5))
#    plt.xticks(np.linspace(0, L, 10, endpoint=True))
    plt.title('$\epsilon$ = {:}, f = {:}, q = {:}'.format(ee, ff[-1], qq), fontsize=12)
    plt.suptitle('Oregonator', fontsize=10)
#    plt.legend(bbox_to_anchor=[1.05, 1.])#, loc=2, borderaxespad=0.)
    plt.tight_layout()

@jit
def f(xy):
    return np.array([(xy[0]*(1-xy[0])-ff*xy[1]*(xy[0]-qq)/(qq+xy[0]))/ee, xy[0] - xy[1]])
#    return np.array([ss * (xy[0] - (xy[0] * xy[0]) - (ff * xy[1] + ww) * (xy[0] - qq) / (xy[0] + qq)), xy[0] - xy[1]])
    # parx = (1 - xy[0] + np.sqrt((1-xy[0])*(1-xy[0]) + 4*qq*xy[0]))/(2*qq)
    # return np.array([(-xy[0] - xy[0]*parx + ff*xy[1])/ss,
                      # ww*(parx-xy[1])])


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
    L = 20                      # space
    Nx = 200                   # num space points
    x = np.linspace(0, L, Nx+1) # mesh points in space
    D = np.array([.1, 0])      # diff coeficient Dx Dy
    T = 10                       # final temperature
    dx = x[1]-x[0]              # space step
#    dt = 0.9*dx*dx/2           # temp step /D[1]
    dt = 0.005
    Nt = int(round(T/dt))       # num temp points
    t = np.linspace(0, T, Nt+1) # mesh points in time
    ksi = 0.5*D*dt/dx**2            # help var
    ee = .02
    ff = 1.5
    ff = 2.5*np.ones(Nx+1)
#    ff[:5] = 1.5
    # ff = -2/(1+np.exp((x-1)/0.25))+3
    qq = 0.02
    # Equation system without diffusion f(x,y)

# initial functions
    initialFunc = np.zeros((2, Nx+1))
#    initialFunc[0] = np.genfromtxt('1.out')*0.5+0.3
#    initialFunc[1] = np.genfromtxt('2.out')+0.3
    initialFunc[0][:5] += 0.001
#    initialFunc[0] = .1/(1+np.exp((x-1)/0.25))
#    initialFunc[1] = 2/(1+np.exp((x-1)/0.25))
#    initialFunc[1] = 1/(4*ff)
#    initialFunc = np.random.rand(2,Nx+1)+1

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
#    ab = trid(Nx, ksi)
    side = np.zeros((Nx+1,2))
    for timeStep in range(Nt):
# right-hand side vector
        runge = RK4(q)
        side[0] = runge[:,0] + ksi*2*(q[:, 1] - q[:, 0])
        for i in range(1,Nx):
            side[i] = runge[:, i] + ksi*(q[:, i-1] - 2*q[:, i] + q[:, i+1])
#        side = np.array([RK4(q[:, i]) + ksi*(q[:, i-1] - 2*q[:, i] + q[:, i+1]) for i in range(Nx)])
# Neumann conditions: derivatives at the edges are null.

#        side = np.append(side, [RK4(q[:, -1]) + ksi*2*(q[:, -2] - q[:, -1])], axis=0)
        side[-1] = runge[:, -1] + ksi*2*(q[:, -2] - q[:, -1])
# Solve the equation a x = b for x, assuming a is banded matrix  using the matrix diagonal ordered form.
#        q = np.array(list(map(lambda x: slin.solve_banded((1, 1), ab[x], side.T[x]), [0, 1])))
        q[0] = slin.solve_banded((1, 1), ab[0], side.T[0])
        q[1] = slin.solve_banded((1, 1), ab[1], side.T[1])
        R.append(q.copy())
        if timeStep % 100 == 0: print(timeStep, "/", Nt, time.strftime("%H:%M:%S", time.localtime()))
        sys.stdout.flush()
    R = np.array(R)
#==============================================================================
#   Print block
#    plt.ioff()
    #plt.ion()
    #plt.close('all')
    plt.style.use('fivethirtyeight')
    fig = plt.figure()
#    fig.set_dpi(100)
    ax1 = fig.add_subplot(1, 1, 1)
    anim = animation.FuncAnimation(fig, animate, range(0, Nt+1,10), interval=10)
    plt.show()
    FFMpegWriter = animation.writers['ffmpeg']
#    metadata = dict(title='1D Fizhugh-Nagumo', artist='Matplotlib', comment='Movie support!')
    writer = FFMpegWriter(fps=30, bitrate=1500)
#    anim.save('{}.mp4'.format(time.strftime("%Y%m%d-%H%M%S")), writer=writer)
#==============================================================================
    print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    toc = time.clock()
    print("%5.3f" % (toc-tic))
