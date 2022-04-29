import numpy as np
from .cost import Cost
from .dynamics import Dynamics
import time


class iLQR():

    def __init__(self, cost: Cost, params):
        self.T = params['T']
        self.N = params['N']

        self.steps = params['max_itr']

        self.tol = 1e-2
        self.lambad = 10
        self.lambad_max = 100
        self.lambad_min = 1e-3

        self.dynamics = Dynamics(params)
        self.alphas = 1.1**(-np.arange(10)**2)

        self.dim_x = self.dynamics.dim_x
        self.dim_u = self.dynamics.dim_u

        self.cost = cost

    def forward_pass(self, nominal_states, nominal_controls, K_closed_loop, k_open_loop, alpha):
        X = np.zeros_like(nominal_states)
        U = np.zeros_like(nominal_controls)

        X[:, 0] = nominal_states[:, 0]
        for t in range(self.N-1):
            K = K_closed_loop[:, :, t]
            k = k_open_loop[:, t]
            u = (nominal_controls[:, t] + alpha * k + K @ (X[:, t] - nominal_states[:, t]))
            X[:, t+1], U[:, t] = self.dynamics.forward_step(X[:, t], u)
        
        # calculate the cost of new trajectory
        J = self.cost.get_cost(X, U)

        return X, U, J

    def backward_pass(self, nominal_states, nominal_controls):
        L_x, L_xx, L_u, L_uu, L_ux = self.cost.get_derivatives(nominal_states, nominal_controls)
        A, B = self.dynamics.get_AB_matrix(nominal_states, nominal_controls)
        reg_mat = self.lambad * np.eye(self.dim_u)
        
        # initialize optimal control
        k_open_loop = np.zeros((self.dim_u, self.N - 1))
        K_closed_loop = np.zeros((self.dim_u, self.dim_x, self.N - 1))
        # set derivatives of value function at final step
        p = L_x[:, -1]
        P = L_xx[:, :, -1]
        for t in range(self.N-2, -1, -1):
            # compute quality function and its derivatives
            Q_x = L_x[:, t] + A[:, :, t].T @ p
            Q_u = L_u[:, t] + B[:, :, t].T @ p
            Q_xx = L_xx[:, :, t] + A[:, :, t].T @ P @ A[:, :, t]
            Q_ux = B[:, :, t].T @ P @ A[:, :, t] + L_ux[:, :, t]
            Q_uu = L_uu[:, :, t] + B[:, :, t].T @ P @ B[:, :, t]

            Q_uu_inv = np.linalg.inv(Q_uu + reg_mat)
            
            # update optimal control
            k_open_loop[:, t] = -Q_uu_inv @ Q_u
            K_closed_loop[:, :, t] = -Q_uu_inv @ Q_ux

            # update derivatives of value function
            p = Q_x - Q_ux.T @ Q_uu_inv @ Q_u
            P = Q_xx - Q_ux.T @ Q_uu_inv @ Q_ux

        return K_closed_loop, k_open_loop

    def solve(self, cur_state, controls=None, obs_list=[], record=False):
        status = 0
        self.lambad = 10

        time0 = time.time()

        if controls is None:
            controls = np.zeros((self.dim_u, self.N))
        states = np.zeros((self.dim_x, self.N))
        states[:, 0] = cur_state

        for i in range(1, self.N):
            states[:,i], _ = self.dynamics.forward_step(states[:, i - 1], controls[:, i - 1])

        self.cost.update_obs(obs_list)

        J = self.cost.get_cost(states, controls)

        converged = False

        # have_not_updated = 0
        for i in range(self.steps):
            K_closed_loop, k_open_loop = self.backward_pass(states, controls)
            updated = False
            for alpha in self.alphas:
                X_new, U_new, J_new = self.forward_pass(states, controls, K_closed_loop, k_open_loop, alpha)
                if J_new <= J:
                    if np.abs((J - J_new) / J) < self.tol:
                        converged = True
                    states, controls, J = X_new, U_new, J_new
                    updated = True
                    break
            if updated:
                self.lambad *= 0.7
            else:
                status = 2
                break
            # self.lambad = min(max(self.lambad_min, self.lambad), self.lambad_max)
            self.lambad = max(self.lambad_min, self.lambad)

            if converged:
                status = 1
                break
        t_process = time.time() - time0
        # print("step, ", i, "alpha:", alpha)

        if record:
            # get parameters for FRS
            K_closed_loop, _ = self.backward_pass(states, controls)
            A, B = self.dynamics.get_AB_matrix(states, controls)
        else:
            K_closed_loop = None
            A, B = None, None
        return states, controls, t_process, status, K_closed_loop, A, B
