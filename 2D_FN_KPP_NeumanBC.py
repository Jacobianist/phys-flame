import time
import numpy as np
import matplotlib.pyplot as plt
import mysolver
import animat

# if __name__ == '__main__':
tic = time.clock()
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
# =============================================================================
L = 10                      # space
Nx = 100                    # space points
Ny = 100
T = 4                       # final time
dt = 0.005                  # time step
Nt = round(T/dt)            # time points
x = np.linspace(0, L, Nx+1) # mesh points in space
y = np.linspace(0, L, Ny+1)
t = np.linspace(0, T, Nt+1) # mesh points in time
dx = x[1] - x[0]            # space step
D = np.array([1, .0])       # diffusion coefficient Dx Dy
ksi = 0.5*D*dt/dx**2        # help var
# =============================================================================
# Initiation of transverse wave
# 1D Fisher-KPP model solver
rangeOf = mysolver.kpp(L, Nx, x, dx, T, dt, Nt, .001)
rangeOf = rangeOf*(1-rangeOf)
# =============================================================================
# initial function
initialFunc = np.zeros((2, Nx+1, Ny+1))+0.1
q = initialFunc.copy()
parA = 1
Qu, Qv = [], [] # all resolve
for iteration in range(Nt):
    A = np.meshgrid(rangeOf[parA], rangeOf[parA])[1]
    A[:, :5] = 0.6-4*A[:, :5]   # reverv var
    A[:, 5:] = 0.6-2*A[:, 5:]   # oscill var
# =============================================================================
    q = mysolver.solveNC(Nx, Ny, dt, ksi, q, A)
# =============================================================================
    Qu.append(q[0])
    Qv.append(q[1])
    parA += 1
    if iteration % 100 == 0: print(iteration,"/",Nt,"||",time.strftime("%H:%M:%S", time.localtime()))
Qu, Qv = np.asarray(Qu), np.asarray(Qv)
# =============================================================================
aniPlot = animat.moveIt(Qu, L, Nt, t, 'fnkppnc')
np.savez('fnkppNC_{}'.format(time.strftime("%Y%m%d-%H%M%S")), Qu)
# =============================================================================
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
toc = time.clock()
print("%5.3f" % (toc-tic))
plt.show()
