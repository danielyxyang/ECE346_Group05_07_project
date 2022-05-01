import numpy as np

class Cost:

    def __init__(self, params):
        self.params = params

    #* New stuff.
    def init_cost(self, frs_list):
        """
        Update soft constraints with list of FRS of dynamic obstacles (?)
        """
        pass

    def get_cost(self, states, controls):
        """
        Calculates the cost given planned states and controls.

        Args:
            states (np.ndarray): 4xN array of planned trajectory.
            controls (np.ndarray): 2xN array of planned control.

        Returns:
            int: scalar cost
        """
        pass

    def get_derivatives(self, states, controls):
        """
        Calculate Jacobian and Hessian of the cost function
        
        Args:
            states: 4xN array of planned trajectory
            controls: 2xN array of planned control
        Returns:
            q, Q, r, R, S
        """
        pass
