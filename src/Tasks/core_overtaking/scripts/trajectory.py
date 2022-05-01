import numpy as np
import itertools
from scipy.spatial.transform import Rotation
from scipy.interpolate import BPoly

class TrajectoryLoop():
    def __init__(self, trajectory):
        self.trajectory = trajectory

        self.ref_v = 1
        self.cur_v = 0
        self.dt = 0.2
        self.track_offset = 0

        self.min_v_threshold = 0.1
    
    def get_trajectory_np(self):
        return np.array([state.state for state in self.trajectory]).T

    def get_reference_trajectory(self, cur_state, min_size=1, ref_accel=1):
        """
        Returns:
            ref_trajectory: (4, N) array with (x, y, v, psi) states and N >= min_size
        """
        # TODO if empty
        if self.ref_v < self.min_v_threshold: # avoiding infinite loop
            return np.repeat([self.trajectory[0].state], min_size, axis=0).T
        
        # interpolate velocity using B-splines and based on reference acceleration
        dv = ref_accel * self.dt
        accel_time = abs(self.ref_v - self.cur_v) / dv
        accel_t = np.arange(self.dt, accel_time, self.dt) # time points during acceleration time
        v_smoothed = BPoly.from_derivatives(
            [0, accel_time], # time of keyframes
            [[self.cur_v, 0], [self.ref_v, 0]], # velocity and derivative keyframes
        )
        v = v_smoothed(accel_t)

        # apply track offset to reference trajectory using line perpendicular to line connecting position at time i-1 and i+1
        rot = np.array([[0, 1], [-1, 0]])
        trajectory = np.array([state.state[0:2] for state in self.trajectory + self.trajectory[0:2]]).T
        normal = rot @ (trajectory[:, 2:] - trajectory[:, :-2])
        ref_trajectory = trajectory[:, 0:-2] + self.track_offset * normal / np.linalg.norm(normal, axis=0)

        # compute closest point on reference trajectory
        distances = np.linalg.norm(ref_trajectory.T - cur_state.state[0:2], axis=1)
        closest_point_index = np.argmin(distances)
        trajectory_loop = itertools.islice(itertools.cycle(ref_trajectory.T), closest_point_index, None)
        
        # define getter function for distance between reference points
        def ref_d(index):
            if index < len(v):
                return v[index] * self.dt
            else:
                return self.ref_v * self.dt

        # obtain trajectory with points spaced according to ref_d(index)
        last_state = next(trajectory_loop)
        ref_trajectory = [last_state]
        next_d_index = 0
        next_d = ref_d(next_d_index)
        for state in trajectory_loop:
            if len(ref_trajectory) >= min_size + 1:
                break

            difference = state - last_state
            distance = np.linalg.norm(difference[0:2])
            
            while next_d <= distance:
                alpha = next_d / distance
                ref_state = (1 - alpha) * last_state + alpha * state
                ref_trajectory.append(ref_state)
                
                next_d_index += 1
                next_d += ref_d(next_d_index)
            next_d -= distance
            last_state = state
        ref_trajectory = np.array(ref_trajectory).T

        return ref_trajectory
    
    def set_reference_velocity(self, ref_v, cur_v, dt):
        self.ref_v = ref_v
        self.cur_v = cur_v
        self.dt = dt
    
    def set_track_offset(self, track_offset):
        self.track_offset = track_offset

    def length(self):
        trajectory_np = self.get_trajectory_np()
        return np.sum(np.linalg.norm(trajectory_np[0:2, 1:] - trajectory_np[0:2, :-1], axis=0))