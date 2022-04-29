import numpy as np
from threading import RLock
from scipy.spatial.transform import Rotation

from MPC import State

class Trajectory():
	def __init__(self, max_list_size=200):
		self.lock = RLock()
		self.max_list_size = max_list_size

		self.trajectory = []

		self.ref_d = 0.2 # spatial separation between reference trajectory points
	
	def add_odom_state(self, odom_msg):
		with self.lock:
			# timestamp
			t = odom_msg.header.stamp
			# position
			x = odom_msg.pose.pose.position.x
			y = odom_msg.pose.pose.position.y
			# orientation
			rot_vec = Rotation.from_quat([
				odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
				odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w,
			]).as_rotvec()
			psi = rot_vec[2]
			# linear velocity
			if self.size():
				dx = x - self.trajectory[-1].state[0]
				dy = y - self.trajectory[-1].state[1]
				dt = (t - self.trajectory[-1].t).to_sec()
				v = np.sqrt(dx * dx + dy * dy) / dt
			else:
				v = 0
			# drop states when reaching max list size
			while self.size() > self.max_list_size - 1:
				self.trajectory.pop(0)
			# add new state to trajectory
			self.trajectory.append(State(np.array([x, y, v, psi]), t))

	def truncate_trajectory(self, x, y):
		with self.lock:
			if self.size() == 0:
				return
			trajectory_np = self.get_trajectory_np()
			current_p = np.array([[x], [y]])
			closest_p_index = np.argmin(np.linalg.norm(trajectory_np[0:2] - current_p, axis=0))
			
			# drop states on trajectory behind closest point
			for _ in range(closest_p_index):
				self.trajectory.pop(0)

	def get_trajectory(self):
		with self.lock:
			return self.trajectory

	def size(self):
		with self.lock:
			return len(self.trajectory)
	
	def get_trajectory_np(self):
		return np.array([state.state for state in self.get_trajectory()]).T

	def get_reference_trajectory(self, min_size=1):
		"""
		Returns:
			ref_trajectory: (4, N) array with (x, y, v, psi) states and N >= min_size
		"""
		trajectory = self.get_trajectory()
		
		# TODO if empty

		# obtain trajectory with points uniformly spaced with distance self.ref_d
		ref_trajectory = [trajectory[0].state]

		distance = 0
		last_state = trajectory[0]
		for state in trajectory[1:]:
			difference = state.state - last_state.state
			distance += np.linalg.norm(difference[0:2])
			while distance >= self.ref_d:
				alpha = self.ref_d / distance
				ref_state = last_state.state + alpha * difference
				ref_trajectory.append(ref_state)
				distance = distance - self.ref_d
			last_state = state
		
		# fill reference trajectory with latest 
		while len(ref_trajectory) < min_size:
			ref_trajectory.append(trajectory[-1].state)

		return np.array(ref_trajectory).T
	
	def set_reference_velocity(self, v, dt):
		self.ref_d = v * dt

	def length(self):
		trajectory_np = self.get_trajectory_np()
		return np.sum(np.linalg.norm(trajectory_np[0:2, 1:] - trajectory_np[0:2, :-1], axis=0))