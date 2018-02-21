import time
import numpy as np
from numba import jit
import numpy.linalg as nlin
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

@jit
def f(xy):
    return np.array([(xy[0]*(1 - xy[0]) - ff*xy[1]*(xy[0] - qq)/(qq + xy[0]))/ee,
                    xy[0] - xy[1]])

@jit
def RK4(xy):
    k1 = dt * f(xy)
    k2 = dt * f(xy + 0.5 * k1)
    k3 = dt * f(xy + 0.5 * k2)
    k4 = dt * f(xy + k3)
    return xy + (k1 + 2 * k2 + 2 * k3 + k4) / 6

@jit
def animate(i):
    ax1.clear()
    plt.plot(x, R[i][0], color='red', label='u')
    plt.plot(x, R[i][1], color='blue', label='v')
    plt.grid(True)
    plt.ylim([np.min(R), np.max(R)])
    plt.xlim([0, L])
    plt.xlabel('time: {:03f}s'.format(t[i]))
    plt.yticks()
    plt.xticks()
    # plt.title('$\epsilon$ = {:}, f = {:}, q = {:}'.format(ee, ff[-1], qq), fontsize=12)
    # plt.suptitle('Oregonator', fontsize=10)
    # plt.legend(bbox_to_anchor=[1.05, 1.])#, loc=2, borderaxespad=0.)
    plt.tight_layout()

tic = time.clock()
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
L = 20                      # space size
dx = .1                     # space step
Nx = round(L/dx)
x = np.linspace(0, L, Nx + 1)  # mesh points in space
T = 5                       # final temperature
dt = .005                   # time step
Nt = round(T/dt)            # time points
t = np.linspace(0, T, Nt + 1)  # mesh points in time
D = np.array([1, .0])       # diffusion coefficient Dx Dy
ksi = 0.5*D*dt/dx*2         # help var

ee = .02
qq = .02
ff = 2*np.ones(Nx+1)

initialFunc = np.zeros((2, Nx+1))
initialFunc[0][:] = np.genfromtxt('zz')#[:,1]
initialFunc[1][:] = np.genfromtxt('xx')#[:,1]
q = initialFunc.copy()  # current resolve
R = []              # all resolve
R.append(q.copy())
ab = np.empty((2, Nx+1, Nx+1))
for i in range(2):
    ab[i] = np.diagflat(np.ones(Nx)*(-ksi[i]), -1) +\
            np.diagflat(np.ones(Nx)*(-ksi[i]), 1) +\
            np.diagflat(np.ones(Nx + 1)*(2*ksi[i] + 1))
    ab[i][0, -1] = -ksi[i]
    ab[i][-1, 0] = -ksi[i]
side = np.zeros((Nx + 1, 2))    # right-hand vector for matrix eq
for timeStep in range(Nt):
    runge = RK4(q)
    side[0] = runge[:, 0] + ksi*(q[:, -1] - 2*q[:, 0] + q[:, 1])
    for i in range(1, Nx):
        side[i] = runge[:, i] + ksi*(q[:, i-1] - 2*q[:, i] + q[:, i+1])
    side[-1] = runge[:, -1] + ksi*(q[:, -2] - 2*q[:, -1] + q[:, 0])
# Solve the equation a x = b for x, assuming a is banded matrix  using the matrix diagonal ordered form.
    q[0] = nlin.solve(ab[0], side.T[0])
    q[1] = nlin.solve(ab[1], side.T[1])
    R.append(q.copy())
R = np.array(R)
plt.close('all')
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
anim = animation.FuncAnimation(fig, animate, range(0, Nt + 1, 10), interval=10)
# =============================================================================
# Animation save block
# FFMpegWriter = animation.writers['ffmpeg']
# metadata = dict(title='1D Oregonator', artist='Matplotlib', comment='Movie support!')
# writer = FFMpegWriter(fps=30, bitrate=1500)
# anim.save('{}.mp4'.format(time.strftime("%Y%m%d-%H%M%S")), writer=writer)
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
toc = time.clock()
print("%5.3f" % (toc - tic))
plt.show()
