#!/usr/bin/env python
import csv, yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import rospy
from nav_msgs.msg import Odometry

from MPC import MPC, State
from iLQR import Track
from cost_trajectory import CostTrajectory
from trajectory import TrajectoryLoop

FRONT_DRIVER = "front_driver"

class FrontDriver():
    def __init__(self):
        rospy.init_node('front_driver_node')
        rospy.loginfo("Start front driver node")
        
        # read parameters
        controller_topic = rospy.get_param("~ControllerTopic")
        odom_topic = rospy.get_param("~PoseTopic")
        params_file = rospy.get_param("~ParamsFile")
        track_file = rospy.get_param("~TrackFile")    

        # load parameters
        with open(params_file) as file:
            self.params = yaml.load(file, Loader=yaml.FullLoader)
        
        self.max_rate_time = 0.01
        self.max_rate_space = 0.025
        
        self.T = self.params['T']
        self.N = self.params['N']
        self.replan_dt = self.T / (self.N - 1)

        # load track file
        x, y = [], []
        with open(track_file, newline='') as f:
            spamreader = csv.reader(f, delimiter=',')
            for i, row in enumerate(spamreader):
                if i > 0:
                    x.append(float(row[0]))
                    y.append(float(row[1]))
        center_line = np.array([x, y])
        self.track = Track(center_line=center_line, width_left=self.params['track_width_L'], width_right=self.params['track_width_R'], loop=True)
        
        # define trajectory based on given track
        self.trajectory = TrajectoryLoop([State(np.array([x, y, 0, 0]), 0) for x, y, in center_line.T])
        self.trajectory.set_track_offset(self.params[FRONT_DRIVER]["track_offset"])

        # subscribe to odometry topic
        rospy.loginfo("Subscribing to {}".format(odom_topic))
        self.sub_odom = rospy.Subscriber(odom_topic, Odometry, self.subscribe_odom, queue_size=1)
        
        self.last_p = None

        # define planner
        self.cost = CostTrajectory(self.params, self._get_ref_traj)
        self.cost.set_mode(FRONT_DRIVER)
        self.planner = MPC(self.cost, self.params, pose_topic=odom_topic, control_topic=controller_topic)
        
        # run planner
        self.planner.run()

    def _get_ref_traj(self, n=None):
        ref_trajectory = self.trajectory.get_reference_trajectory(self.last_p, min_size=n, ref_accel=self.params[FRONT_DRIVER]["ref_accel"])
        if n is not None:
            return ref_trajectory[:, :n]
        else:
            return ref_trajectory

    def _odom_to_state(self, odom_msg, prev_state=None):
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
        if prev_state is not None:
            dx = x - prev_state.state[0]
            dy = y - prev_state.state[1]
            dt = (t - prev_state.t).to_sec()
            v = np.sqrt(dx * dx + dy * dy) / dt
        else:
            v = 0
        # return state
        return State(np.array([x, y, v, psi]), t)

    def subscribe_odom(self, odom_msg):
        # return state
        self.last_p = self._odom_to_state(odom_msg, prev_state=self.last_p)
        self.trajectory.set_reference_velocity(self.params[FRONT_DRIVER]["ref_vel"], self.last_p.state[2], self.replan_dt)

if __name__ == '__main__':
    overtaker = FrontDriver()
    rospy.spin()
