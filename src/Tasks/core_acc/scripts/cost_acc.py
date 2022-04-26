import numpy as np
from iLQR import Cost, Track

class CostACC(Cost):

  def __init__(self, params, get_ref_traj):
    self.params = params
    self.get_ref_traj = get_ref_traj

    # load parameters
    self.T = params['T']  # Planning Time Horizon
    self.N = params['N']  # number of planning steps
    self.dt = self.T / (self.N - 1)  # time step for each planning step
    
    self.v_max = params['v_max']  # max velocity
    self.wheelbase = params['wheelbase']

    # cost
    self.w_vel = params['w_vel']
    # self.w_contour = params['w_contour']
    self.w_theta = params['w_theta']
    self.w_accel = params['w_accel']
    self.w_delta = params['w_delta']

    # self.track_offset = params['track_offset']

    self.W_state = np.array([[self.w_contour, 0], [0, self.w_vel]])
    self.W_control = np.array([[self.w_accel, 0], [0, self.w_delta]])

    # useful constants
    self.zeros = np.zeros((self.N))
    self.ones = np.ones((self.N))
  
  #* New stuff.
  def update_obs(self, frs_list):
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
    closest_pt, slope, theta = self.get_ref_traj().get_closest_pts(states[:2, :])

    transform = np.array([
      [np.sin(slope), -np.cos(slope), self.zeros, self.zeros],
      [self.zeros, self.zeros, self.ones, self.zeros]
    ])

    ref_states = np.zeros_like(states)
    ref_states[0, :] = closest_pt[0, :] + np.sin(slope) * self.track_offset
    ref_states[1, :] = closest_pt[1, :] - np.cos(slope) * self.track_offset
    ref_states[2, :] = self.v_max

    error = states - ref_states 
    Q_trans = np.einsum('abn, bcn->acn', np.einsum('dan, ab -> dbn', transform.transpose(1, 0, 2), self.W_state), transform)

    c_state = np.einsum('an, an->n', error, np.einsum('abn, bn->an', Q_trans, error))
    c_control = np.einsum('an, an->n', controls, np.einsum('ab, bn->an', self.W_control, controls))
    c_progress = -self.w_theta * theta  #(theta[-1] - theta[0])
    
    # constraints
    # c_constraint = self.soft_constraints.get_cost(
    #     states, controls, closest_pt, slope
    # )    

    J = np.sum(c_state + c_control + c_progress)
    # J = np.sum(c_state + c_constraint + c_control + c_progress)

    return J

  def get_derivatives(self, states, controls):
    """
    Calculate Jacobian and Hessian of the cost function
    
    Args:
      states: 4xN array of planned trajectory
      controls: 2xN array of planned control
    Returns:
      q, Q, r, R, S
    """
    closest_pt, slope, theta = self.get_ref_traj().get_closest_pts(states[:2, :])

    # c_x_cons, c_xx_cons, c_u_cons, c_uu_cons, c_ux_cons = (
    #     self.soft_constraints.get_derivatives(
    #         states, controls, closest_pt, slope
    #     )
    # )

    c_x_cost, c_xx_cost = self._get_cost_state_derivative(
        states, closest_pt, slope
    )

    c_u_cost, c_uu_cost = self._get_cost_control_derivative(controls)

    q = c_x_cost
    Q = c_xx_cost

    r = c_u_cost
    R = c_uu_cost

    S = c_ux_cons

    return q, Q, r, R, S
  
  def _get_cost_state_derivative(self, states, closest_pt, slope):
    '''
    Calculate Jacobian and Hessian of the cost function with respect to state
        states: 4xN array of planned trajectory
        closest_pt: 2xN array of each state's closest point [x,y] on the track
        slope: 1xN array of track's slopes (rad) at closest points
    '''
    transform = np.array([[
        np.sin(slope), -np.cos(slope), self.zeros, self.zeros
    ], [self.zeros, self.zeros, self.ones, self.zeros]])
    ref_states = np.zeros_like(states)
    ref_states[0, :] = closest_pt[0, :]+np.sin(slope) * self.track_offset
    ref_states[1, :] = closest_pt[1, :]-np.cos(slope) * self.track_offset
    ref_states[2, :] = self.v_max

    error = states - ref_states 
    Q_trans = np.einsum(
        'abn, bcn->acn',
        np.einsum(
            'dan, ab -> dbn', transform.transpose(1, 0, 2), self.W_state
        ), transform
    )- self.track_offset

    # shape [4xN]
    c_x = 2 * np.einsum('abn, bn->an', Q_trans, error)

    c_x_progress = -self.w_theta * np.array([
        np.cos(slope), np.sin(slope), self.zeros, self.zeros
    ])
    c_x = c_x + c_x_progress
    c_xx = 2 * Q_trans

    return c_x, c_xx

  def _get_cost_control_derivative(self, controls):
    '''
    Calculate Jacobian and Hessian of the cost function w.r.t the control
        controls: 2xN array of planned control
    '''
    c_u = 2 * np.einsum('ab, bn->an', self.W_control, controls)
    c_uu = 2 * np.repeat(self.W_control[:, :, np.newaxis], self.N, axis=2)
    return c_u, c_uu
