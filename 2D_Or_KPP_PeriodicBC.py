import time
import numpy as np
import matplotlib.pyplot as plt
import mysolver
import animat

# if __name__ == '__main__':
tic = time.clock()
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
# =============================================================================
L = 20                      # space size
Nx = 200                    # space points
Ny = 200
T = 10                      # final time
dt = 0.0025                 # time step
Nt = round(T/dt)            # time points
x = np.linspace(0, L, Nx+1) # mesh points in space
y = np.linspace(0, L, Ny+1)
t = np.linspace(0, T, Nt+1) # mesh points in time
dx = x[1] - x[0]            # space step
D = np.array([1, 0])        # diffusion coefficient Dx Dy
ksi = 0.5*D*dt/dx**2        # help var
# =============================================================================
# Initiation of transverse wave
# 1D Fisher-KPP model solver
rangeOf = mysolver.kpp(L, Nx, x, dx, T, dt, Nt, .1)
rangeOf = rangeOf*(1-rangeOf)
rangeOf = 4-7*rangeOf
# =============================================================================
# Parameters setup for oregonator
ee = 0.02
#ff = 2.1
qq = 0.02
# initial function
initialFunc = np.zeros((2, Nx+1, Ny+1))
initialFunc[0] = np.genfromtxt('orU')#[::2]
initialFunc[1] = np.genfromtxt('orV')#[::2]
q = initialFunc.copy()
parA = 400
Qu, Qv = [], []             # all resolve

for iteration in range(Nt):
    ff = np.meshgrid(rangeOf[parA], rangeOf[parA])[1]
# =============================================================================
    q = mysolver.solvePCO(Nx, Ny, dt, ksi, q, ee, qq, ff)
# =============================================================================
    Qu.append(q[0])
    Qv.append(q[1])
    parA += 1
    if iteration % 100 == 0: print(iteration, "/", Nt, "||", time.strftime("%H:%M:%S", time.localtime()))
Qu, Qv = np.asarray(Qu), np.asarray(Qv)
# =============================================================================
aniPlot = animat.moveIt(Qu, L, Nt, t, 'orkppPC')
# aniPlot = animat.moveIt(Qu, L, Nt, t)
# plt.show()
# =============================================================================
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
toc = time.clock()
print("%5.3f" % (toc-tic))
