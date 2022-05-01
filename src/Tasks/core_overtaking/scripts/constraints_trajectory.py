from typing import List
import numpy as np
from iLQR import EllipsoidObj, state2ell
import time

class ConstraintsTrajectory:

    def __init__(self, params):
        self.params = params

        # load parameters
        self.T = params['T']  # Planning Time Horizon
        self.N = params['N']  # number of planning steps
        self.dt = self.T / (self.N - 1)  # time step for each planning step

        self.wheelbase = params['wheelbase']  # vehicle chassis length
        # self.delta_min = params['delta_min']  # min steering angle rad
        # self.delta_max = params['delta_max']  # max steering angle rad
        # self.a_min = params['a_min']  # min longitudial accel
        # self.a_max = params['a_max']  # max longitudial accel
        self.alat_max = params['alat_max']  # max lateral accel
        self.alat_min = -params['alat_max']  # min lateral accel

        # parameter for barrier functions
        self.q1_lat = 1
        self.q2_lat = 5
        self.q1_obs = 1
        self.q2_obs = 5

        # useful constants
        self.zeros = np.zeros((self.N))
        self.ones = np.ones((self.N))

        self.gamma = 0.9
        """ ego system"""
        ell_a = self.params["length"] / 2.0
        ell_b = self.params["width"] / 2.0
        wheelbase = self.params["wheelbase"]
        self.ego = EllipsoidObj(
            q=np.array([wheelbase / 2, 0])[:, np.newaxis],
            Q=np.diag([ell_a ** 2, ell_b ** 2]),
        )
        self.ego_ell = None
        """ obstacles represented as FRS"""
        self.obs_list = None

    def init_constraints(self, obs_list):
        self.obs_list = obs_list
    
    def set_mode(self, mode):
        self.q1_lat = self.params[mode]['q1_lat']
        self.q2_lat = self.params[mode]['q2_lat']

        self.q1_obs = self.params[mode]['q1_obs']
        self.q2_obs = self.params[mode]['q2_obs']


    def get_cost(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        # Lateral Acceleration constraint
        accel = states[2, :]**2 * np.tan(controls[1, :]) / self.wheelbase
        error_ub = accel - self.alat_max
        error_lb = self.alat_min - accel

        b_ub = self.q1_lat * (np.exp(self.q2_lat * error_ub) - 1)
        b_lb = self.q1_lat * (np.exp(self.q2_lat * error_lb) - 1)
        c_lat = b_lb + b_ub

        # Obstacle constraints
        c_obs = np.zeros(self.N)
        if len(self.obs_list)>0:
            # Patch footprint around state trajectory
            self.ego_ell = [state2ell(states[:, i], self.ego) for i in range(self.N)]
            for i in range(self.N):
                ego_i = self.ego_ell[i]
                for obs_j in self.obs_list:  # obs_j is a list of obstacles.
                    obs_j_i = obs_j[i]  # Get the ith obstacle in list obs_j.
                    c_obs[i] += self.gamma**i * ego_i.obstacle_cost(
                        obs_j_i, self.q1_obs, self.q2_obs)

        return c_lat + c_obs

    def get_derivatives(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """
        Calculates the Jacobian and Hessian of soft constraint cost.

        Args:
            states: 4xN array of planned trajectory
            controls: 2xN array of planned control
        """

        # lateral acceleration constraints
        c_x_lat, c_xx_lat, c_u_lat, c_uu_lat, c_ux_lat = self._lat_accec_bound_derivative(states, controls)

        # obstacle constraints
        c_x_obs = np.zeros_like(c_x_lat)
        c_xx_obs = np.zeros_like(c_xx_lat)
        if len(self.obs_list)>0:
            for i in range(self.N):
                ego_i = self.ego_ell[i]
                for obs_j in self.obs_list:
                    obs_j_i = obs_j[i]
                    c_x_obs_temp, c_xx_obs_temp = ego_i.obstacle_derivative(states[:, i], self.wheelbase / 2, obs_j_i, self.q1_obs, self.q2_obs)
                    c_x_obs[:, i] += self.gamma**i * c_x_obs_temp
                    c_xx_obs[:, :, i] += self.gamma**i * c_xx_obs_temp

        # sum up
        c_x_cons = c_x_lat + c_x_obs
        c_xx_cons = c_xx_lat + c_xx_obs

        c_u_cons = c_u_lat
        c_uu_cons = c_uu_lat
        c_ux_cons = c_ux_lat

        return c_x_cons, c_xx_cons, c_u_cons, c_uu_cons, c_ux_cons

    def _lat_accec_bound_derivative(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """
        Calculates the Jacobian and Hessian of Lateral Acceleration soft constraint
            cost.

        Args:
            states: 4xN array of planned trajectory
            controls: 2xN array of planned control
        """
        c_x = np.zeros((4, self.N))
        c_xx = np.zeros((4, 4, self.N))
        c_u = np.zeros((2, self.N))
        c_uu = np.zeros((2, 2, self.N))
        c_ux = np.zeros((2, 4, self.N))

        # calculate the acceleration
        accel = states[2, :]**2 * np.tan(controls[1, :]) / self.wheelbase

        error_ub = accel - self.alat_max
        error_lb = self.alat_min - accel

        b_ub = self.q1_lat * np.exp(np.clip(self.q2_lat * error_ub, None, 20))
        b_lb = self.q1_lat * np.exp(np.clip(self.q2_lat * error_lb, None, 20))

        da_dx = 2 * states[2, :] * np.tan(controls[1, :]) / self.wheelbase
        da_dxx = 2 * np.tan(controls[1, :]) / self.wheelbase

        da_du = states[2, :]**2 / (np.cos(controls[1, :])**2 * self.wheelbase)
        da_duu = (states[2, :]**2 * np.sin(controls[1, :]) / (np.cos(controls[1, :])**3 * self.wheelbase))

        da_dux = 2 * states[2, :] / (np.cos(controls[1, :])**2 * self.wheelbase)

        c_x[2, :] = self.q2_lat * (b_ub - b_lb) * da_dx
        c_u[1, :] = self.q2_lat * (b_ub - b_lb) * da_du

        c_xx[2, 2, :] = self.q2_lat**2 * (b_ub + b_lb) * da_dx**2 + self.q2_lat * (b_ub - b_lb) * da_dxx
        c_uu[1, 1, :] = self.q2_lat**2 * (b_ub + b_lb) * da_du**2 + self.q2_lat * (b_ub - b_lb) * da_duu

        c_ux[1, 2, :] = (self.q2_lat**2 * (b_ub + b_lb) * da_dx * da_du + self.q2_lat * (b_ub - b_lb) * da_dux)
        return c_x, c_xx, c_u, c_uu, c_ux
