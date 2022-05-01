import numpy as np
from threading import RLock
from scipy.spatial.transform import Rotation
from scipy.interpolate import BPoly

from MPC import State

class Trajectory():
    def __init__(self, max_list_size=200, fixed=-1):
        self.lock = RLock()
        self.max_list_size = max_list_size
        self.fixed = fixed

        self.trajectory = []

        self.ref_v = 1 # spatial separation between reference trajectory points (of host car)
        self.cur_v = 1 # spatial separation between reference trajectory points (of this car)
        self.dt = 0.2

        self.min_v_threshold = 0.1
    
    def _odom_to_state(self, odom_msg, prev_index=-1):
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
            if prev_index is not None and self.size() > 0:
                dx = x - self.trajectory[prev_index].state[0]
                dy = y - self.trajectory[prev_index].state[1]
                dt = (t - self.trajectory[prev_index].t).to_sec()
                v = np.sqrt(dx * dx + dy * dy) / dt
            else:
                v = 0
            # return state
            return State(np.array([x, y, v, psi]), t)

    def add_odom_state(self, odom_msg):
        with self.lock:
            state = self._odom_to_state(odom_msg)
            # drop states when reaching max list size
            while self.size() > self.max_list_size - 1:
                self.trajectory.pop(self.fixed + 1)
            # add new state to trajectory
            self.trajectory.append(state)
    
    
    def update_odom_state(self, odom_msg, index):
        with self.lock:
            # update state in trajectory
            state = self._odom_to_state(odom_msg, prev_index=index - 1 if index > 0 else None)
            while index >= self.size(): # if list too short pad with given state
                self.trajectory.append(state)
            self.trajectory[index] = state

    def truncate_trajectory(self, x, y):
        """
        Truncate trajectory while ignoring fixed data points
        """
        with self.lock:
            if self.size() <= self.fixed + 1:
                return
            
            trajectory_np = self.get_trajectory_np()[:, self.fixed + 1:]
            current_p = np.array([[x], [y]])
            closest_p_index = np.argmin(np.linalg.norm(trajectory_np[0:2] - current_p, axis=0))
            
            # drop states on trajectory behind closest point
            for _ in range(closest_p_index):
                self.trajectory.pop(self.fixed + 1)

    def get_trajectory(self):
        with self.lock:
            return self.trajectory

    def size(self):
        with self.lock:
            return len(self.trajectory)
    
    def get_trajectory_np(self):
        return np.array([state.state for state in self.get_trajectory()]).T

    def get_reference_trajectory(self, min_size=1, ref_accel=1, prefix=[]):
        """
        Returns:
            ref_trajectory: (4, N) array with (x, y, v, psi) states and N >= min_size
        """
        trajectory = prefix + self.get_trajectory()
        
        # TODO if empty
        if self.ref_v < self.min_v_threshold: # avoiding infinite loop
            return np.repeat([trajectory[0].state], min_size, axis=0).T
        
        # interpolate velocity using B-splines and based on reference acceleration
        dv = ref_accel * self.dt
        accel_time = abs(self.ref_v - self.cur_v) / dv
        accel_t = np.arange(self.dt, accel_time, self.dt) # time points during acceleration time
        v_smoothed = BPoly.from_derivatives(
            [0, accel_time], # time of keyframes
            [[self.cur_v, 0], [self.ref_v, 0]], # velocity and derivative keyframes
        )
        v = v_smoothed(accel_t)

        # define getter function for distance between reference points
        def ref_d(index):
            if index < len(v):
                return v[index] * self.dt
            else:
                return self.ref_v * self.dt

        # obtain trajectory with points spaced according to ref_d(index)
        ref_trajectory = [trajectory[0].state]
        last_state = trajectory[0]
        next_d_index = 0
        next_d = ref_d(next_d_index)
        for state in trajectory[1:]:
            difference = state.state - last_state.state
            distance = np.linalg.norm(difference[0:2])
            
            while next_d <= distance:
                alpha = next_d / distance
                ref_state = (1 - alpha) * last_state.state + alpha * state.state
                ref_trajectory.append(ref_state)
                
                next_d_index += 1
                next_d += ref_d(next_d_index)
            next_d -= distance
            last_state = state
        
        # fill reference trajectory with latest state
        while len(ref_trajectory) < min_size:
            ref_trajectory.append(trajectory[-1].state)

        return np.array(ref_trajectory).T
    
    def set_reference_velocity(self, ref_v, cur_v, dt):
        self.ref_v = ref_v
        self.cur_v = cur_v
        self.dt = dt

    def length(self):
        trajectory_np = self.get_trajectory_np()
        return np.sum(np.linalg.norm(trajectory_np[0:2, 1:] - trajectory_np[0:2, :-1], axis=0))