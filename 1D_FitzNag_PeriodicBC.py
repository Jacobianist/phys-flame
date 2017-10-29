import time
import sys
import numpy as np
from numba import jit
# import scipy.linalg as slin
import numpy.linalg as nlin
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns


# FitzHugh–Nagumo model function
@jit
def f(xy):
    return np.array([250*(-xy[0]*(xy[0] - A)*(xy[0] - 1) - xy[1]), 
                        xy[0] - xy[1]])


# Runge Kutta 4th
@jit
def RK4(xy):
    k1 = dt * f(xy)
    k2 = dt * f(xy + 0.5*k1)
    k3 = dt * f(xy + 0.5*k2)
    k4 = dt * f(xy + k3)
    return xy + (k1 + 2*k2 + 2*k3 + k4)/6


# The plot shows the wave evolving with time
@jit
def animate(i):
    ax1.clear()
    plt.plot(x, R[i][0], color='red', label='u')
    plt.plot(x, R[i][1], color='blue', label='v')
    plt.grid(True)
    plt.ylim([np.min(R), np.max(R)])
    plt.xlim([0, L])
    plt.xlabel('time: {:03f}s'.format(t[i]))
    # plt.yticks()
    # plt.xticks()
    # plt.title('dx = {:.1e}, dt = {:.1e}'.format(dx, dt))
    # plt.legend(bbox_to_anchor=[1.05, 1.])#, loc=2, borderaxespad=0.)
    plt.tight_layout()


# if __name__ == '__main__':
tic = time.clock()
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
# =============================================================================
# vars
L = 10                      # space
Nx = 200                    # space points
x = np.linspace(0, L, Nx+1) # mesh points in space
dx = x[1] - x[0]            # space step
T = 2                       # final temperature
dt = 0.005                  # time step
Nt = round(T/dt)            # time points
t = np.linspace(0, T, Nt+1) # mesh points in time
D = np.array([1, .0])       # diffusion coefficient Dx Dy
ksi = 0.5*D*dt/dx**2        # help var

# initial functions
initialFunc = np.zeros((2, Nx+1))
# initialFunc[0][:5] += 0.001
initialFunc[0] = np.genfromtxt('1.out')
initialFunc[1] = np.genfromtxt('2.out')
# initialFunc[0] = 1/(1+np.exp((x*5-1)/0.25))
# initialFunc = np.random.rand(2,Nx+1)+0.1
# =============================================================================

q = initialFunc[:]  # current resolve
R = []              # all resolve
R.append(q.copy())
# Tridiagonal matrix for Crank–Nicolson method to banded matrix for scipy.linalg.solve_banded()
# with PERIODIC CONDITIONS
ab = np.empty((2, Nx+1, Nx+1))
for i in range(2):
    ab[i] = np.diagflat(np.ones(Nx)*(-ksi[i]), -1) +\
            np.diagflat(np.ones(Nx)*(-ksi[i]), 1) +\
            np.diagflat(np.ones(Nx + 1)*(2*ksi[i] + 1))
    ab[i][0, -1] = -ksi[i]
    ab[i][-1, 0] = -ksi[i]
side = np.zeros((Nx + 1, 2))    # right-hand side vector for matrix eq
A = 0.1 # FHN main parameter
for timeStep in range(Nt):
    runge = RK4(q)
# with PERIODIC CONDITIONS
    side[0] = runge[:, 0] + ksi*(q[:, -1] - 2*q[:, 0] + q[:, 1])
    for i in range(1, Nx):
        side[i] = runge[:, i] + ksi*(q[:, i-1] - 2*q[:, i] + q[:, i+1])
    side[-1] = runge[:, -1] + ksi*(q[:, -2] - 2*q[:, -1] + q[:, 0])
# Solve the equation a x = b for x, assuming a is banded matrix  using the matrix diagonal ordered form.
    q[0] = nlin.solve(ab[0], side.T[0])
    q[1] = nlin.solve(ab[1], side.T[1])
    R.append(q.copy())
    if timeStep % 100 == 0: print(timeStep, "/", Nt, time.strftime("%H:%M:%S", time.localtime()))
R = np.array(R)
# =============================================================================
# Plot block
# plt.ioff()
# plt.ion()
plt.close('all')
plt.style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
anim = animation.FuncAnimation(fig, animate, range(0, Nt+1), interval=10)
# =============================================================================
# Animation save block
# FFMpegWriter = animation.writers['ffmpeg']
# metadata = dict(title='1D Fizhugh-Nagumo', artist='Matplotlib', comment='Movie support!')
# writer = FFMpegWriter(fps=30, bitrate=1500)
# anim.save('{}.mp4'.format(time.strftime("%Y%m%d-%H%M%S")), writer=writer)
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
toc = time.clock()
print("%5.3f" % (toc-tic))

plt.show()
