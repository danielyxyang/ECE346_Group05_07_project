import numpy as np
from threading import Lock

class Trajectory():
    def __init__(self, max_list_size=200):
        self.max_list_size = max_list_size
        self.x_traj = []
        self.y_traj = []

        self.x_traj_mid = []
        self.y_traj_mid = []

        self.lock = Lock()
    
    def add_point(self, x, y):
        with self.lock:
            while len(self.x_traj) > self.max_list_size - 1:
                self.x_traj.pop(0)
                self.y_traj.pop(0)
                self.x_traj_mid.pop(0)
                self.y_traj_mid.pop(0)
            
            self.x_traj.append(x)
            self.y_traj.append(y)
            
            if len(self.x_traj) > 1:
                self.x_traj_mid.append((self.x_traj[-1] + self.x_traj[-2]) / 2.0)
                self.y_traj_mid.append((self.y_traj[-1] + self.y_traj[-2]) / 2.0)
            else:
                self.x_traj_mid.append(self.x_traj[-1])
                self.y_traj_mid.append(self.y_traj[-1])

    
    def truncate_trajectory(self, x, y):
        with self.lock:
            cur_pos = np.array([x, y])
            traj_mid = np.array([self.x_traj_mid, self.y_traj_mid]).T
            midpoint_difference = traj_mid - cur_pos
            midpoint_distances = np.sum(midpoint_difference ** 2, axis=1)
            closest_midpoint = np.argmin(midpoint_distances)

            self.x_traj = self.x_traj[closest_midpoint:]
            self.y_traj = self.y_traj[closest_midpoint:]
            self.x_traj_mid = self.x_traj_mid[closest_midpoint:]
            self.y_traj_mid = self.y_traj_mid[closest_midpoint:]
    
    def get_trajectory(self):
        with self.lock:
            return self.x_traj, self.y_traj, self.x_traj_mid, self.y_traj_mid
    
    def is_available(self):
        return len(self.x_traj) > 0