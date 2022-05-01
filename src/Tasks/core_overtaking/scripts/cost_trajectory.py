import numpy as np
from iLQR import Cost
from constraints_trajectory import ConstraintsTrajectory

class CostTrajectory(Cost):

    def __init__(self, params, get_ref_traj):
        self.params = params
        self.get_ref_traj = get_ref_traj
        self.soft_constraints = ConstraintsTrajectory(params)
        self.gamma = 0.9

        # load parameters
        self.T = params['T']  # Planning Time Horizon
        self.N = params['N']  # number of planning steps
        self.dt = self.T / (self.N - 1)  # time step for each planning step
        
        # cost
        self.w_ref_traj = 1
        self.w_accel = 1
        self.w_delta = 1
        self.W_control = np.array([[self.w_accel, 0], [0, self.w_delta]])

        # useful constants
        self.zeros = np.zeros((self.N))
        self.ones = np.ones((self.N))
        self.gammas = np.cumprod(np.full((self.N), fill_value=self.gamma))
        self.gammas_inv = np.cumprod(np.full((self.N), fill_value=self.gamma))[::-1]
    
    #* New stuff.
    def init_cost(self, obs_list):
        """
        Update soft constraints with list of FRS of dynamic obstacles (?)
        """
        self.ref_traj = self.get_ref_traj(n=self.N)

        self.obs_list = obs_list
        self.soft_constraints.init_constraints(obs_list)
    
    def set_mode(self, mode):
        self.w_ref_traj = self.params[mode]['w_ref_traj']
        self.w_accel = self.params[mode]['w_accel']
        self.w_delta = self.params[mode]['w_delta']
        self.W_control = np.array([[self.w_accel, 0], [0, self.w_delta]])
        
        self.soft_constraints.set_mode(mode)

    def get_cost(self, states, controls):
        """
        Calculates the cost given planned states and controls.

        Args:
            states (np.ndarray): 4xN array of planned trajectory.
            controls (np.ndarray): 2xN array of planned control.

        Returns:
            int: scalar cost
        """
        # initialize total cost
        J_vec = np.zeros(self.N)

        # position penalty
        c_ref_traj = self.w_ref_traj * np.sum((self.ref_traj[0:2] - states[0:2]) ** 2, axis=0)
        J_vec += c_ref_traj

        # control regularizer
        c_control = np.einsum("an,ab,bn->n", controls, self.W_control, controls)
        J_vec += c_control

        # constraints
        c_constraint = self.soft_constraints.get_cost(states, controls)
        J_vec += c_constraint
        
        # total scalar cost
        J = np.sum(J_vec)
        return J

    def get_derivatives(self, states, controls):
        """
        Calculate Jacobian and Hessian of the cost function
        
        Args:
            states: 4xN array of planned trajectory
            controls: 2xN array of planned control
        Returns:
            c_x, c_xx, c_u, c_uu, c_ux
        """
        # initialize derivatives
        c_x = np.zeros(shape=(4, self.N))
        c_xx = np.zeros(shape=(4, 4, self.N))
        c_u = np.zeros(shape=(2, self.N))
        c_uu = np.zeros(shape=(2, 2, self.N))
        c_ux = np.zeros(shape=(2, 4, self.N))

        if len(self.ref_traj) == 0:
            return c_x, c_xx, c_u, c_uu, c_ux

        # position penalty
        c_x += self.w_ref_traj * np.concatenate([
            -2 * (self.ref_traj[0:2] - states[0:2]),
            [self.zeros, self.zeros],
        ])
        c_xx += self.w_ref_traj * np.array([
            [np.full(self.N, 2), self.zeros, self.zeros, self.zeros],
            [self.zeros, np.full(self.N, 2), self.zeros, self.zeros],
            [self.zeros, self.zeros, self.zeros, self.zeros],
            [self.zeros, self.zeros, self.zeros, self.zeros],
        ])
        # c_xx += self.w_ref_traj * np.repeat(np.array([
        #     [2, 0, 0, 0],
        #     [0, 2, 0, 0],
        #     [0, 0, 0, 0],
        #     [0, 0, 0, 0],
        # ])[:, :, np.newaxis], self.N, axis=2)

        # control regularizer
        c_u += 2 * np.einsum("ab,bn->n", self.W_control, controls)
        c_uu += 2 * np.repeat(self.W_control[:, :, np.newaxis], self.N, axis=2)

        # constraints
        c_x_cons, c_xx_cons, c_u_cons, c_uu_cons, c_ux_cons = self.soft_constraints.get_derivatives(states, controls)
        c_x += c_x_cons
        c_xx += c_xx_cons
        c_u += c_u_cons
        c_uu += c_uu_cons
        c_ux += c_ux_cons

        return c_x, c_xx, c_u, c_uu, c_ux