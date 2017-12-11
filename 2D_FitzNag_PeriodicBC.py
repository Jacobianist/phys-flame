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
# 1D FitzHugh–Nagumo model solver
rangeOf = mysolver.fn(L, Nx, x, dx, T, dt, Nt, 0.3)
rangeOf = rangeOf[:, 0, :]
rangeOf = 0.65 - 0.7*rangeOf # modulation
# =============================================================================
# initial function
initialFunc = np.zeros((2, Nx+1, Ny+1))
initialFunc[0] = np.genfromtxt('1.out')[::2]
initialFunc[1] = np.genfromtxt('2.out')[::2]
q = initialFunc.copy()
parA = 1
Qu, Qv = [], [] # all resolve

for iteration in range(Nt):
    A = np.meshgrid(rangeOf[parA], rangeOf[parA])[1]
# =============================================================================
    q = mysolver.solvePC(Nx, Ny, dt, ksi, q, A)
# =============================================================================
    Qu.append(q[0])
    Qv.append(q[1])
    parA += 1
#    if iteration % 100 == 0: print(iteration,"/",Nt,"||",time.strftime("%H:%M:%S", time.localtime()))
Qu, Qv = np.asarray(Qu), np.asarray(Qv)
# =============================================================================
aniPlot = animat.moveIt(Qu, L, Nt, t, 'fnfnpc')
np.savez('fnfnPC_{}'.format(time.strftime("%Y%m%d-%H%M%S")), Qu)
# =============================================================================
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
toc = time.clock()
print("%5.3f" % (toc-tic))
plt.show()
