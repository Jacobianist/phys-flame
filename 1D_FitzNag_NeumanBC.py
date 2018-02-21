import time
import sys
import numpy as np
from numba import jit
import scipy.linalg as slin
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import seaborn as sns


# Runge Kutta 4th
@jit
def RK4(xy):
    k1 = dt * f(xy)
    k2 = dt * f(xy + 0.5 * k1)
    k3 = dt * f(xy + 0.5 * k2)
    k4 = dt * f(xy + k3)
    return xy + (k1 + 2 * k2 + 2 * k3 + k4) / 6


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


# Tridiagonal matrix for Crank–Nicolson method to banded matrix for
# scipy.linalg.solve_banded()
def trid(Nx, ksi):
    a = np.ones(Nx + 1) * (-ksi[:, np.newaxis])    # above main diag
    c = a.copy()                # under main diagonal
    a[:, 0], c[:, -1] = 0, 0
    a[:, 1] = -2 * ksi
    c[:, -2] = -2 * ksi
    b = np.ones(Nx + 1) * (2 * ksi[:, np.newaxis] + 1)    # main diag
    return np.array([np.vstack((a[i], b[i], c[i])) for i in [0, 1]])

# FitzHugh–Nagumo model function
@jit
def f(xy):
    return 0.5*np.array([E*(-xy[0]*(xy[0] - A)*(xy[0] - 1) - xy[1]),
                        xy[0] - xy[1]])

# if __name__ == '__main__':
tic = time.clock()
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
# =============================================================================
# vars
L = 20                      # space
Nx = 200                    # space points
x = np.linspace(0, L, Nx + 1)  # mesh points in space
dx = x[1] - x[0]            # space step
T = 5                       # final time
dt = 0.005                  # time step
Nt = round(T / dt)          # time points
t = np.linspace(0, T, Nt + 1)  # mesh points in time
D = np.array([.5, .0])       # diffusion coefficient Dx Dy
ksi = 0.5 * D * dt / dx**2  # help var
E = 100
# initial functions
initialFunc = np.zeros((2, Nx + 1))# + 0.1
# initialFunc[0][:5] += 0.001
initialFunc[0] = 1/(1+np.exp((x-1)/0.25))
# initialFunc = np.random.rand(2,Nx+1)+0.1
# =============================================================================

q = initialFunc[:]  # current resolve
R = []              # all resolve
R.append(q.copy())
ab = trid(Nx, ksi)  # with NEUMANN CONDITIONS
side = np.zeros((Nx + 1, 2))    # right-hand vector for matrix eq
A = 0.1 * np.ones(Nx + 1)  # FHN main parameter
A[:5] = -0.2        # Reverb
for timeStep in range(Nt):
    runge = RK4(q)
# Neumann conditions: derivatives at the edges are null.
    side[0] = runge[:, 0] + ksi * 2 * (q[:, 1] - q[:, 0])
    for i in range(1, Nx):
        side[i] = runge[:, i] + ksi * (q[:, i - 1] - 2 * q[:, i] + q[:, i + 1])
        # side = np.array([RK4(q[:, i]) + ksi*(q[:, i-1] - 2*q[:, i] + q[:, i+1]) for i in range(Nx)])
    # side = np.append(side, [RK4(q[:, -1]) + ksi*2*(q[:, -2] - q[:, -1])], axis=0)
    side[-1] = runge[:, -1] + ksi * 2 * (q[:, -2] - q[:, -1])
# Solve the equation a x = b for x, assuming a is banded matrix  using the matrix diagonal ordered form.
#        q = np.array(list(map(lambda x: slin.solve_banded((1, 1), ab[x], side.T[x]), [0, 1])))
    q[0] = slin.solve_banded((1, 1), ab[0], side.T[0])
    q[1] = slin.solve_banded((1, 1), ab[1], side.T[1])
    R.append(q.copy())
    if timeStep % 100 == 0: print(timeStep, "/", Nt, time.strftime("%H:%M:%S", time.localtime()))
R = np.array(R)
#R =  0.5 - 0.5*R
# =============================================================================
# Plot block
# plt.ioff()
# plt.ion()
plt.close('all')
#plt.style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
anim = animation.FuncAnimation(fig, animate, range(0, Nt + 1, 5), interval=10)
# =============================================================================
# Animation save block
# FFMpegWriter = animation.writers['ffmpeg']
# metadata = dict(title='1D Fizhugh-Nagumo', artist='Matplotlib', comment='Movie support!')
# writer = FFMpegWriter(fps=30, bitrate=1500)
# anim.save('{}.mp4'.format(time.strftime("%Y%m%d-%H%M%S")), writer=writer)
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
toc = time.clock()
print("%5.3f" % (toc - tic))
plt.show()
