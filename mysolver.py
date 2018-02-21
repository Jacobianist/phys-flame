import numpy as np
from numba import jit
import scipy.linalg as slin
import numpy.linalg as nlin
from scipy.integrate import odeint, ode
# =============================================================================
# 1D FitzHugh–Nagumo model
def fn(L, Nx, x, dx, T, dt, Nt, e, w=1.):
    D = np.array([w, .0])           # diff coeficient Dx Dy
    ksi = 0.5*D*dt/dx**2            # help var
    initialFunc = np.zeros((2, Nx+1))
    initialFunc[0] = 1/(1+np.exp((x-1)/0.25))
    q = initialFunc[:]
    R = []
    R.append(q.copy())
    a = np.ones(Nx+1)*(-ksi[:, np.newaxis])    # above main diag
    c = a.copy()                # under main diagonal
    a[:, 0], c[:, -1] = 0, 0
    a[:, 1] = -2.*ksi
    c[:, -2] = -2.*ksi
    b = np.ones(Nx+1)*(2*ksi[:, np.newaxis] + 1)    # main diag
    ab = np.array([np.vstack((a[i], b[i], c[i])) for i in [0, 1]])     # banded matrix for solve_banded()
    side = np.zeros((Nx+1, 2))
    @jit
    def f1(xy):
        return np.array([(-xy[0]*(xy[0]-A)*(xy[0]-1) - xy[1])*e,
                           xy[0] - xy[1]])
    @jit
    def RK(xy):
        k1 = dt * f1(xy)
        k2 = dt * f1(xy + 0.5*k1)
        k3 = dt * f1(xy + 0.5*k2)
        k4 = dt * f1(xy + k3)
        return (xy + (k1 + 2*k2 + 2*k3 + k4)/6)
    for timeStep in range(2*Nt):
        A = 0.1
        runge = RK(q)
        side[0] = runge[:, 0] + ksi*2*(q[:, 1] - q[:, 0])
        for i in range(1, Nx):
            side[i] = runge[:, i] + ksi*(q[:, i-1] - 2*q[:, i] + q[:, i+1])
        side[-1] = runge[:, -1] + ksi*2*(q[:, -2] - q[:, -1])
        q = np.array(list(map(lambda x: slin.solve_banded((1, 1), ab[x], side.T[x]), [0, 1])))
        R.append(q.copy())
    R = np.array(R)
    return R

# =============================================================================
# 1D Fisher-KPP model
def kpp(L, Nx, x, dx, T, dt, Nt, D=1.):
    ksi = 0.5*D*dt/dx**2            # help var
    initialFunc = 1/(1+np.exp((x-4)/1.0))
#    initialFunc = np.zeros(Nx+1)
#    initialFunc[:10] += 0.1
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
    @jit
    def f1(xy):
        return xy*(1-xy)
    @jit
    def RK(xy):
        k1 = dt * f1(xy)
        k2 = dt * f1(xy + 0.5*k1)
        k3 = dt * f1(xy + 0.5*k2)
        k4 = dt * f1(xy + k3)
        return (xy + (k1 + 2*k2 + 2*k3 + k4)/6)
    for timeStep in range(4*Nt):
        runge = RK(q)
        side[0] = runge[0] + ksi*2*(q[1] - q[0])
        for i in range(1, Nx):
            side[i] = runge[i] + ksi*(q[i-1] - 2*q[i] + q[i+1])
        side[-1] = runge[-1] + ksi*2*(q[-2] - q[-1])
        q = slin.solve_banded((1, 1), ab, side)
        R.append(q.copy())
    R = np.array(R)
    return R

def tridNC(Ny, ksi):
    a = np.ones(Ny+1)*(-ksi)    # above main diag
    c = a.copy()                # under main diagonal
    a[:, 0], c[:, -1] = 0, 0
    a[:, 1] = -2.*ksi.ravel()
    c[:, -2] = -2.*ksi.ravel()
    b = np.ones(Ny+1)*(2*ksi[:, np.newaxis] + 1)    # main diag
    return np.array([np.vstack((a[i], b[i], c[i])) for i in [0, 1]])     # banded matrix for solve_banded()


# =============================================================================
# one iteration of alternating direction implicit method
# with NEUMANN CONDITIONS
# FitzHugh–Nagumo
def solveNC(Nx, Ny, dt, ksi, q, A):
    def f(xy):
        return np.array([500*(-xy[0]*(xy[0] - A)*(xy[0] - 1) - xy[1]),
                            xy[0] - xy[1]])
    # Runge Kutta 4th
    def RK4(xy):
        k1 = dt * f(xy)
        k2 = dt * f(xy + 0.5 * k1)
        k3 = dt * f(xy + 0.5 * k2)
        k4 = dt * f(xy + k3)
        return xy + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    ab = tridNC(Nx, ksi[:, np.newaxis]) # with NEUMANN CONDITIONS
    # Tridiagonal matrix for Crank–Nicolson method to banded matrix for
    # scipy.linalg.solve_banded()
    runge = RK4(q)
    side0, side1 = np.zeros((2, Nx+1, Ny+1))
    for i in range(Nx+1):
        for j in range(1, Ny):
            side0[i,j], side1[i,j] = ksi*(q[:,i,j-1] - 2*q[:,i,j] + q[:,i,j+1]) + runge[:,i,j]
        side0[i,0], side1[i,0] = ksi*2*(q[:,i,1] - q[:,i,0]) + runge[:,i,0]
        side0[i,-1], side1[i,-1] = ksi*2*(q[:,i,-2] - q[:,i,-1]) + runge[:,i,-1]
    q0 = slin.solve_banded((1, 1), ab[0], side0)
    q1 = slin.solve_banded((1, 1), ab[1], side1)
    q = np.array((q0, q1))

    for j in range(Ny+1):
        for i in range(1,Nx):
            side0[i,j], side1[i,j] = ksi*(q[:,i-1,j] - 2*q[:,i,j] + q[:,i+1,j]) + q[:,i,j]
        side0[0,j], side1[0,j] = ksi*2*(q[:,1,j] - q[:,0,j]) + q[:,0,j]
        side0[-1,j], side1[-1,j] = ksi*2*(q[:,-2,j] - q[:,-1,j]) + q[:,-1,j]
    q0 = slin.solve_banded((1, 1), ab[0], side0)
    q1 = slin.solve_banded((1, 1), ab[1], side1)
    q = np.array((q0, q1))
    return q

# =============================================================================
# one iteration of alternating direction implicit method
# with PERIODIC CONDITIONS
# FitzHugh–Nagumo
def solvePC(Nx, Ny, dt, ksi, q, A):
    def f(xy):
        return np.array([500*(-xy[0]*(xy[0] - A)*(xy[0] - 1) - xy[1]),
                            xy[0] - xy[1]])
    # Runge Kutta 4th
    def RK4(xy):
        k1 = dt * f(xy)
        k2 = dt * f(xy + 0.5 * k1)
        k3 = dt * f(xy + 0.5 * k2)
        k4 = dt * f(xy + k3)
        return xy + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    ab = np.empty((2, Nx+1, Nx+1))
    for i in range(2):
        ab[i] = np.diagflat(np.ones(Nx)*(-ksi[i]),-1) +\
                np.diagflat(np.ones(Nx+1)*(2*ksi[i] + 1)) +\
                np.diagflat(np.ones(Nx)*(-ksi[i]), 1)
        ab[i][0,-1] = -ksi[i]
        ab[i][-1,0] = -ksi[i]
    runge = RK4(q)
    side0, side1 = np.zeros((2, Nx+1, Ny+1))
    for i in range(Nx+1):
        for j in range(1, Ny):
            side0[i,j], side1[i,j] = ksi*(q[:,i,j-1] - 2*q[:,i,j] + q[:,i,j+1]) + runge[:,i,j]
        side0[i,0], side1[i,0] = ksi*(q[:,i,-1] - 2*q[:,i,0] + q[:,i,1]) + runge[:,i,0]
        side0[i,-1], side1[i,-1] = ksi*(q[:,i,-2] - 2*q[:,i,-1] + q[:,i,0]) + runge[:,i,-1]
    q0 = nlin.solve(ab[0], side0)
    q1 = nlin.solve(ab[1], side1)
    q = np.array((q0, q1))

    for j in range(Ny+1):
        for i in range(1,Nx):
            side0[i,j], side1[i,j] = ksi*(q[:,i-1,j] - 2*q[:,i,j] + q[:,i+1,j]) + q[:,i,j]
        side0[0,j], side1[0,j] = ksi*(q[:,-1,j] - 2*q[:,0,j] + q[:,1,j]) + q[:,0,j]
        side0[-1,j], side1[-1,j] = ksi*(q[:,-2,j] - 2*q[:,-1,j] + q[:,0,j]) + q[:,-1,j]
    q0 = nlin.solve(ab[0], side0)
    q1 = nlin.solve(ab[1], side1)
    q = np.array((q0, q1))
    return q

# =============================================================================
# one iteration of alternating direction implicit method
# with NEUMANN CONDITIONS
# Oregonator
def solveNCO(Nx, Ny, dt, ksi, q, ee, qq, ff):
    def f(xy):
        return np.array([(xy[0]*(1-xy[0])-ff*xy[1]*(xy[0]-qq)/(qq+xy[0]))/ee, xy[0] - xy[1]])
    def RK4(xy):
        k1 = dt * f(xy)
        k2 = dt * f(xy + 0.5*k1)
        k3 = dt * f(xy + 0.5*k2)
        k4 = dt * f(xy + k3)
        return xy + (k1 + 2*k2 + 2*k3 + k4)/6
    ab = tridNC(Nx, ksi[:, np.newaxis])
    # with NEUMANN CONDITIONS
    # Tridiagonal matrix for Crank–Nicolson method to banded matrix for
    # scipy.linalg.solve_banded()
    runge = RK4(q)
    side0, side1 = np.zeros((2, Nx+1, Ny+1))
    for i in range(Nx+1):
        for j in range(1, Ny):
            side0[i,j], side1[i,j] = ksi*(q[:,i,j-1] - 2*q[:,i,j] + q[:,i,j+1]) + runge[:,i,j]
        side0[i,0], side1[i,0] = ksi*2*(q[:,i,1] - q[:,i,0]) + runge[:,i,0]
        side0[i,-1], side1[i,-1] = ksi*2*(q[:,i,-2] - q[:,i,-1]) + runge[:,i,-1]
    q0 = slin.solve_banded((1, 1), ab[0], side0)
    q1 = slin.solve_banded((1, 1), ab[1], side1)
    q = np.array((q0, q1))
    for j in range(Ny+1):
        for i in range(1,Nx):
            side0[i,j], side1[i,j] = ksi*(q[:,i-1,j] - 2*q[:,i,j] + q[:,i+1,j]) + q[:,i,j]
        side0[0,j], side1[0,j] = ksi*2*(q[:,1,j] - q[:,0,j]) + q[:,0,j]
        side0[-1,j], side1[-1,j] = ksi*2*(q[:,-2,j] - q[:,-1,j]) + q[:,-1,j]
    q0 = slin.solve_banded((1, 1), ab[0], side0)
    q1 = slin.solve_banded((1, 1), ab[1], side1)
    q = np.array((q0, q1))
    return q

# =============================================================================
# one iteration of alternating direction implicit method
# with PERIODIC CONDITIONS
# Oregonator
def solvePCO(Nx, Ny, dt, ksi, q, ee, qq, ff):
    def f(xy):
        return np.array([(xy[0]*(1-xy[0])-ff*xy[1]*(xy[0]-qq)/(qq+xy[0]))/ee, xy[0] - xy[1]])
    # Runge Kutta 4th
    def RK4(xy):
        k1 = dt * f(xy)
        k2 = dt * f(xy + 0.5 * k1)
        k3 = dt * f(xy + 0.5 * k2)
        k4 = dt * f(xy + k3)
        return xy + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    ab = np.empty((2, Nx+1, Nx+1))
    for i in range(2):
        ab[i] = np.diagflat(np.ones(Nx)*(-ksi[i]),-1) +\
                np.diagflat(np.ones(Nx+1)*(2*ksi[i] + 1)) +\
                np.diagflat(np.ones(Nx)*(-ksi[i]), 1)
        ab[i][0,-1] = -ksi[i]
        ab[i][-1,0] = -ksi[i]
    runge = RK4(q)
    side0, side1 = np.zeros((2, Nx+1, Ny+1))
    for i in range(Nx+1):
        for j in range(1, Ny):
            side0[i,j], side1[i,j] = ksi*(q[:,i,j-1] - 2*q[:,i,j] + q[:,i,j+1]) + runge[:,i,j]
        side0[i,0], side1[i,0] = ksi*(q[:,i,-1] - 2*q[:,i,0] + q[:,i,1]) + runge[:,i,0]
        side0[i,-1], side1[i,-1] = ksi*(q[:,i,-2] - 2*q[:,i,-1] + q[:,i,0]) + runge[:,i,-1]
    q0 = nlin.solve(ab[0], side0)
    q1 = nlin.solve(ab[1], side1)
    q = np.array((q0, q1))
    for j in range(Ny+1):
        for i in range(1,Nx):
            side0[i,j], side1[i,j] = ksi*(q[:,i-1,j] - 2*q[:,i,j] + q[:,i+1,j]) + q[:,i,j]
        side0[0,j], side1[0,j] = ksi*(q[:,-1,j] - 2*q[:,0,j] + q[:,1,j]) + q[:,0,j]
        side0[-1,j], side1[-1,j] = ksi*(q[:,-2,j] - 2*q[:,-1,j] + q[:,0,j]) + q[:,-1,j]
    q0 = nlin.solve(ab[0], side0)
    q1 = nlin.solve(ab[1], side1)
    q = np.array((q0, q1))
    return q
