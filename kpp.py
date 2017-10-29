import time
import numpy as np
from numba import jit
import numpy.linalg as nlin
import scipy.linalg as slin
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
import seaborn as sns

tic = time.clock()
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
L = 10
T = 5
Nx = 100
x = np.linspace(0, L, Nx+1)     # mesh points in space
D = .1           # diff coeficient Dx Dy
dx = x[1]-x[0]                  # space step
dt = 0.005
Nt = int(round(2*T/dt))         # num 2x temp points
t = np.linspace(0,T,Nt+1)
ksi = 0.5*D*dt/dx**2            # help var
initialFunc = np.zeros(Nx+1)
initialFunc[30:50] += 0.1
# initialFunc = 1/(1+np.exp((x-1)/0.25))
q = initialFunc[:]
R = []
R.append(q.copy())
a = np.ones(Nx+1)*(-ksi)    # above main diag
c = a.copy()                # under main diagonal
a[0], c[-1] = 0, 0
a[1] = -2.*ksi
c[-2] = -2.*ksi
b = np.ones(Nx+1)*(2*ksi + 1)    # main diag
ab = np.vstack((a, b, c))     # banded matrix for solve_banded()
side = np.zeros(Nx+1)

# @jit
def f1(xy):
    return 5*xy*(1-xy)

# @jit
def RK(xy):
    k1 = dt * f1(xy)
    k2 = dt * f1(xy + 0.5*k1)
    k3 = dt * f1(xy + 0.5*k2)
    k4 = dt * f1(xy + k3)
    return (xy + (k1 + 2*k2 + 2*k3 + k4)/6)

for timeStep in range(Nt):
    runge = RK(q)
    side[0] = runge[0] + ksi*2*(q[1] - q[0])
    for i in range(1, Nx):
        side[i] = runge[i] + ksi*(q[i-1] - 2*q[i] + q[i+1])
    side[-1] = runge[-1] + ksi*2*(q[-2] - q[-1])
    q = slin.solve_banded((1, 1), ab, side)
    R.append(q.copy())
R = np.array(R)


def animate(i):
# The plot shows the temperature evolving with time
    ax1.clear()
    plt.plot(x, R[i], color='red', label='u')#+str(round(t[i], 2))+'s')
    plt.grid(True)
    plt.ylim([np.min(R), np.max(R)])
    plt.xlim([0, L])
    plt.xlabel('time: {:03f}s'.format(t[i]))
    plt.tight_layout()

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
anim = animation.FuncAnimation(fig, animate, range(0, Nt+1,10), interval=10)
plt.show()
###############################################################################
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
toc = time.clock()
print("%5.3f" % (toc-tic))
