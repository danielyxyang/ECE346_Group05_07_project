import numpy as np
from iLQR import Cost
from constraints_acc import ConstraintsACC

class CostACC(Cost):

	def __init__(self, params, get_ref_traj):
		self.params = params
		self.get_ref_traj = get_ref_traj
		self.gamma = 0.9
		# self.soft_constraints = ConstraintsACC(params)

		# load parameters
		self.T = params['T']  # Planning Time Horizon
		self.N = params['N']  # number of planning steps
		self.dt = self.T / (self.N - 1)  # time step for each planning step
		
		self.v_max = params['v_max']  # max velocity
		self.wheelbase = params['wheelbase']

		# cost
		self.w_vel = params['w_vel']
		self.w_contour = params['w_contour']
		self.w_theta = params['w_theta']
		self.w_accel = params['w_accel']
		self.w_delta = params['w_delta']

		self.track_offset = params['track_offset']

		self.W_state = np.array([[self.w_contour, 0], [0, self.w_vel]])
		self.W_control = np.array([[self.w_accel, 0], [0, self.w_delta]])

		# useful constants
		self.zeros = np.zeros((self.N))
		self.ones = np.ones((self.N))
		self.gammas = np.cumprod(np.full((self.N), fill_value=self.gamma))
	
	#* New stuff.
	def init_cost(self, frs_list):
		"""
		Update soft constraints with list of FRS of dynamic obstacles (?)
		"""
		x_traj, y_traj, x_traj_mid, y_traj_mid = self.get_ref_traj()
		self.ref_traj = np.array([x_traj, y_traj]).T
		self.ref_traj_midpoints = np.array([x_traj_mid, y_traj_mid]).T
		self.ref_traj_distances = np.cumsum(np.sum((self.ref_traj[1:] - self.ref_traj[:-1]) ** 2, axis=1)[::-1])[::-1]
		self.ref_traj_distances = np.append(self.ref_traj_distances, 0)
		print(self.ref_traj.shape, self.ref_traj_midpoints.shape, self.ref_traj_distances.shape)
		# if len(self.ref_traj) > 1:
		# 	ref_traj_extended = np.concatenate((
		# 		[self.ref_traj[0] - (self.ref_traj[1] - self.ref_traj[0])],
		# 		self.ref_traj,
		# 	))
		# 	self.ref_traj_midpoints = (ref_traj_extended[1:] + ref_traj_extended[:-1]) / 2
		# else:
		# 	self.ref_traj_midpoints = self.ref_traj
		
		# print(self.ref_traj)
		# print(self.ref_traj_midpoints)

	def get_cost(self, states, controls):
		"""
		Calculates the cost given planned states and controls.

		Args:
			states (np.ndarray): 4xN array of planned trajectory.
			controls (np.ndarray): 2xN array of planned control.

		Returns:
			int: scalar cost
		"""
		if len(self.ref_traj_midpoints) == 0:
			return 1 # avoid division by zero

		# initialize total cost
		J_vec = 0

		# state regularizer
		# error_vel = states[2] - self.v_max
		# c_state = self.w_vel * (error_vel ** 2)
		# J_vec += c_state
		
		# control regularizer
		# c_control = np.einsum("an,ab,bn->n", controls, self.W_control, controls)
		# J_vec += c_control
		
		# progress reward
		closest_midpoint_distance = np.zeros(self.N)
		for i in range(self.N):
			midpoint_difference = self.ref_traj_midpoints - states[0:1, i]
			midpoint_distances = np.sum(midpoint_difference ** 2, axis=1)
			closest_midpoint = np.argmin(midpoint_distances)
			closest_midpoint_distance[i] = midpoint_distances[closest_midpoint] + self.ref_traj_distances[closest_midpoint]
		c_progress = self.w_theta * closest_midpoint_distance
		print(c_progress)
		J_vec += c_progress

		# total scalar cost
		# J = np.sum(self.gammas * J_vec)
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

		if len(self.ref_traj_midpoints) == 0:
			return c_x, c_xx, c_u, c_uu, c_ux

		# state derivatives
		# error_vel = states[2] - self.v_max
		# c_x += np.array([self.zeros, self.zeros, self.w_vel * (2 * error_vel), self.zeros])

		# control regularizer
		# c_u += 2 * np.einsum("ab,bn->n", self.W_control, controls)
		# c_uu += 2 * np.repeat(self.W_control[:, :, np.newaxis], self.N, axis=2)

		# progress derivatives
		closest_midpoint_difference = np.zeros(shape=(2, self.N))
		for i in range(self.N):
			midpoint_difference = self.ref_traj_midpoints - states[0:1, i]
			midpoint_distances = np.sum(midpoint_difference ** 2, axis=1) * 100
			closest_midpoint = np.argmin(midpoint_distances)
			closest_midpoint_difference[:, i] = midpoint_difference[closest_midpoint]
		c_x += np.concatenate((-2 * self.w_theta * closest_midpoint_difference, [self.zeros, self.zeros]))

		return c_x, c_xx, c_u, c_uu, c_ux
	
	# def _get_cost_state_derivative(self, states, closest_pt, slope):
	# 	"""
	# 	Calculate Jacobian and Hessian of the cost function with respect to state
	# 			states: 4xN array of planned trajectory
	# 			closest_pt: 2xN array of each state's closest point [x,y] on the track
	# 			slope: 1xN array of track's slopes (rad) at closest points
	# 	"""
	# 	transform = np.array([
	# 		[np.sin(slope), -np.cos(slope), self.zeros, self.zeros],
	# 		[self.zeros, self.zeros, self.ones, self.zeros],
	# 	])
	# 	ref_states = np.zeros_like(states)
	# 	ref_states[0, :] = closest_pt[0, :] + np.sin(slope) * self.track_offset
	# 	ref_states[1, :] = closest_pt[1, :] - np.cos(slope) * self.track_offset
	# 	ref_states[2, :] = self.v_max
	# 	error = states - ref_states 
	# 	Q_trans = np.einsum('abn, bcn->acn', np.einsum('dan, ab -> dbn', transform.transpose(1, 0, 2), self.W_state), transform) - self.track_offset


	# 	c_x = 2 * np.einsum('abn, bn->an', Q_trans, error)
	# 	c_xx = 2 * Q_trans

	# 	return c_x
	
	# def _get_cost_progress_derivative(self, slope):
	# 	c_x = -self.w_theta * np.array([np.cos(slope), np.sin(slope), self.zeros, self.zeros])
	# 	return c_x

	# def _get_cost_control_derivative(self, controls):
	# 	"""
	# 	Calculate Jacobian and Hessian of the cost function w.r.t the control
	# 			controls: 2xN array of planned control
	# 	"""
	# 	c_u = 2 * np.einsum('ab, bn->an', self.W_control, controls)
	# 	c_uu = 2 * np.repeat(self.W_control[:, :, np.newaxis], self.N, axis=2)
	# 	return c_u, c_uu
