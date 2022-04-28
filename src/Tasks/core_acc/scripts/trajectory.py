import numpy as np
from threading import Lock
from scipy.spatial.transform import Rotation

from MPC import State

class Trajectory():
    def __init__(self, min_list_size=0, max_list_size=200):
        self.min_list_size = min_list_size
        self.max_list_size = max_list_size

        # self.t = [0] * min_list_size
        # self.x = [0] * min_list_size
        # self.y = [0] * min_list_size
        # self.psi = [0] * min_list_size
        # self.v = [0] * min_list_size
        
        self.t = []
        self.x = []
        self.y = []
        self.psi = []
        self.v = []
        # self.trajectory = [State(np.zeros(4), 0)] * min_list_size # TODO

        self.lock = Lock()
    
    def add_odom_state(self, odom_msg):
        # timestamp
        t = odom_msg.header.stamp

        # position
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        
        # pose
        rot_vec = Rotation.from_quat([
            odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w,
        ]).as_rotvec()
        psi = rot_vec[2]

        with self.lock:
            # linear velocity
            if len(self.x) > 0:
                dx = x - self.x[-1]
                dy = y - self.y[-1]
                dt = (t - self.t[-1]).to_sec()
                v = np.sqrt(dx * dx + dy * dy) / dt
            else:
                v = 0
            
            while len(self.x) > self.max_list_size - 1:
                self.t.pop(0)
                self.x.pop(0)
                self.y.pop(0)
                self.psi.pop(0)
                self.v.pop(0)
            
            self.t.append(t)
            self.x.append(x)
            self.y.append(y)
            self.psi.append(psi)
            self.v.append(v)
            
    def truncate_trajectory(self, x, y):
        with self.lock:
            cur_pos = np.array([x, y])
            trajectory = np.array([self.x, self.y]).T
            
            closest_state_index = np.argmin(np.sum((trajectory - cur_pos) ** 2, axis=1))
            closest_state_index = min(closest_state_index, len(self.x) - self.min_list_size)
            for _ in range(closest_state_index):
                self.t.pop(0)
                self.x.pop(0)
                self.y.pop(0)
                self.psi.pop(0)
                self.v.pop(0)
    
    def get_trajectory(self):
        with self.lock:
            return self.x, self.y, self.psi, self.v
    
    def is_available(self):
        return len(self.x) > 0