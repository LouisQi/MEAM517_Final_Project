import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import cholesky
from math import sin, cos
import math
from scipy.interpolate import interp1d
from scipy.integrate import ode
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.linalg import solve_continuous_are

from pydrake.solvers import MathematicalProgram, Solve, OsqpSolver
import pydrake.symbolic as sym

from pydrake.all import MonomialBasis, OddDegreeMonomialBasis, Variables


class Quadrotor(object):
  def __init__(self, Q, R, Qf):
    self.g = 9.81
    self.m = 1
    self.a = 0.25
    self.Ix = 8.1 * 1e-3
    self.Iy = 8.1 * 1e-3
    self.Iz = 14.2 * 1e-3
    self.Q = Q
    self.R = R
    self.Qf = Qf

    # Input limits
    self.umin = 0
    self.umax = 5.5

    self.n_x = 6
    self.n_u = 2
   
  def x_d(self):
    # Nominal state
    return np.array([0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0])

  def u_d(self):
    # Nominal input
    return np.array([self.m*self.g,0,0,0])

  def continuous_time_full_dynamics(self, x, u):
    # Dynamics for the quadrotor
    g = self.g
    m = self.m
    a = self.a
    Ix = self.Ix
    Iy = self.Iy
    Iz = self.Iz

    x1, x2, y1, y2, z1, z2, phi1, phi2, theta1, theta2, psi1, psi2 = x.reshape(-1).tolist()
    ft, tau_x, tau_y, tau_z = u.reshape(-1).tolist()
    
    xdot = np.array([x2,
                    ft/m*(np.sin(phi1)*np.sin(psi1)+np.cos(phi1)*np.cos(psi1)*np.sin(theta1)),
                    y2,
                    ft/m*(np.cos(phi1)*np.sin(psi1)*np.sin(theta1)-np.cos(psi1)*np.sin(phi1)),
                    z2,
                    -g+ft/m*np.cos(phi1)*np.cos(theta1),
                    phi2,
                    a*(Iy-Iz)/Ix*theta2*psi2+tau_x/Ix,
                    theta2,
                    a*(Iz-Ix)/Iy*phi2*psi2+tau_y/Iy,
                    psi2,
                    a*(Ix-Iy)/Iz*phi2*theta2+tau_z/Iz])
    return xdot

  def continuous_time_linearized_dynamics(self):
    # Dynamics linearized at the fixed point
    # This function returns A and B matrix
    A = np.zeros((12,12))
    A[:3, 3:6] = np.identity(3)
    A[-3:,6:9] = np.identity(3)
    A[6,1] = -self.g
    A[7,0] = self.g

    B = np.zeros((12,4))
    B[3,1] = 1/self.Ix
    B[4,2] = 1/self.Iy
    B[5,3] = 1/self.Iz
    B[8,0] = 1/self.m 

    return A, B

  def discrete_time_linearized_dynamics(self, T):
    # Discrete time version of the linearized dynamics at the fixed point
    # This function returns A and B matrix of the discrete time dynamics
    A_c, B_c = self.continuous_time_linearized_dynamics()
    A_d = np.identity(12) + A_c * T
    B_d = B_c * T

    return A_d, B_d

  def add_initial_state_constraint(self, prog, x, x_current):
    # TODO: impose initial state constraint.
    # Use AddBoundingBoxConstraint
    prog.AddBoundingBoxConstraint(x_current, x_current, x[0])
    # pass

  def add_input_saturation_constraint(self, prog, x, u, N):
    # TODO: impose input limit constraint.
    # Use AddBoundingBoxConstraint
    # The limits are available through self.umin and self.umax
    for i in range (N-1):
        prog.AddBoundingBoxConstraint(self.umin - self.u_d(), self.umax - self.u_d(), u[i])
    # pass

  def add_dynamics_constraint(self, prog, x, u, N, T):
    # TODO: impose dynamics constraint.
    # Use AddLinearEqualityConstraint(expr, value)
    A, B = self.discrete_time_linearized_dynamics(T)
    for i in range (N-1):
        prog.AddLinearEqualityConstraint(np.array(A@x[i]+B@u[i]-x[i+1]),np.zeros(12))

    # pass

  def add_cost(self, prog, x, u, N):
    # TODO: add cost.
    cost = 0

    for i in range (N):
        cost += x[i].T @ self.Q @ x[i]

    prog.AddQuadraticCost(cost)
    # pass

  def compute_mpc_feedback(self, x_current, use_clf=False):
    '''
    This function computes the MPC controller input u
    '''

    # Parameters for the QP
    N = 10
    T = 0.1

    # Initialize mathematical program and declare decision variables
    prog = MathematicalProgram()
    x = np.zeros((N, 12), dtype="object")  # Corrected dimensions for x
    for i in range(N):
        x[i] = prog.NewContinuousVariables(12, "x_" + str(i))  # Corrected number of continuous variables for x
    u = np.zeros((N-1, 4), dtype="object")  # Corrected dimensions for u
    for i in range(N-1):
        u[i] = prog.NewContinuousVariables(4, "u_" + str(i))  # Corrected number of continuous variables for u

    # Add constraints and cost
    self.add_initial_state_constraint(prog, x, x_current)
    self.add_input_saturation_constraint(prog, x, u, N)
    self.add_dynamics_constraint(prog, x, u, N, T)
    self.add_cost(prog, x, u, N)

    # Solve the QP
    solver = OsqpSolver()
    result = solver.Solve(prog)

    u_mpc = np.zeros(4)  # Corrected dimensions for u_mpc
    # Retrieve the controller input from the solution of the optimization problem
    if result.is_success():
        u_mpc = result.GetSolution(u[0])
        u_mpc += self.u_d()

    return u_mpc

  def compute_lqr_feedback(self, x):
    '''
    Infinite horizon LQR controller
    '''
    A, B = self.continuous_time_linearized_dynamics()
    S = solve_continuous_are(A, B, self.Q, self.R)
    K = -inv(self.R) @ B.T @ S
    u = self.u_d() + K @ x
    return u
