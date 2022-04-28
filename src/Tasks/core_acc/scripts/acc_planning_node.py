#!/usr/bin/env python
import csv, yaml
import numpy as np
import matplotlib.pyplot as plt

import rospy
from std_msgs.msg import String
from nav_msgs.msg import Odometry

from iLQR import Track
from MPC import MPC
from cost_acc import CostACC
from trajectory import Trajectory


MAX_LIST_SIZE = 100

class PoseSubscriber:
    def __init__(self):
        rospy.init_node('acc_planning_node')
        rospy.loginfo("Start ACC planning node")

        # read arguments
        controller_topic = rospy.get_param("~ControllerTopic")
        pose_topic = rospy.get_param("~PoseTopic")
        pose_host_topic = rospy.get_param("~PoseHostTopic")
        params_file = rospy.get_param("~PlanParamsFile")
        track_file = rospy.get_param("~TrackFile")

        # load parameters
        with open(params_file) as file:
            self.params = yaml.load(file, Loader=yaml.FullLoader)
        
        self.T = self.params['T']
        self.N = self.params['N']
        self.replan_dt = self.T / (self.N - 1)
        
        # load track for plotting
        x, y = [], []
        with open(track_file, newline='') as f:
            spamreader = csv.reader(f, delimiter=',')
            for i, row in enumerate(spamreader):
                if i > 0:
                    x.append(float(row[0]))
                    y.append(float(row[1]))
        self.track = Track(center_line=np.array([x, y]), width_left=self.params['track_width_L'], width_right=self.params['track_width_R'], loop=True)

        # initialize trajectory
        self.traj = Trajectory(max_list_size=1)
        self.traj_host = Trajectory(min_list_size=self.N, max_list_size=MAX_LIST_SIZE)
        self.last_t = None

        # create pose subscriber
        rospy.loginfo("Subscribing to {}".format(pose_topic))
        self.sub_pose = rospy.Subscriber(pose_topic, Odometry, self.subscribe_pose, queue_size=1)
        rospy.loginfo("Subscribing to {}".format(pose_host_topic))
        self.sub_pose_host = rospy.Subscriber(pose_host_topic, Odometry, self.subscribe_pose_host, queue_size=1)

        def _get_ref_traj(n=None):
            x, y, psi, v = self.traj_host.get_trajectory()
            if n is not None:
                return x[:n], y[:n], psi[:n], v[:n]
            else:
                return x, y, psi, v
        
        self.cost = CostACC(self.params, _get_ref_traj)
        self.planner = MPC(self.cost, self.params, pose_topic=pose_topic, control_topic=controller_topic)

    def subscribe_pose(self, poseMsg):
        self.traj.add_odom_state(poseMsg)

    def subscribe_pose_host(self, odomMsg):
        if self.last_t is not None and (odomMsg.header.stamp - self.last_t).to_sec() < self.replan_dt:
            return
        
        self.traj_host.add_odom_state(odomMsg)

        if self.traj.is_available():
            x, y, _, _ = self.traj.get_trajectory()
            self.traj_host.truncate_trajectory(x[0], y[0])
        
        self.last_t = odomMsg.header.stamp
        # with self.lock_host:
            # if not self.traj_init:
            #   if self.current_pos:
            #     self.x_traj = list(np.linspace(self.x, poseMsg.pose.position.x, MAX_LIST_SIZE))
            #     self.y_traj = list(np.linspace(self.y, poseMsg.pose.position.y, MAX_LIST_SIZE))
            #     self.traj_init = True
            #     rospy.loginfo("Trajectory initialized")
            #   else:
            #     return

    def plot_pose(self):
        self.track.plot_track()
        self.track.plot_track_center()

        x, y, _, _ = self.traj.get_trajectory()
        plt.scatter(x, y, c="orange", marker="*")
        
        x, y, _, _ = self.traj_host.get_trajectory()
        plt.scatter(x[-1:], y[-1:], c="green", marker="*")
        plt.scatter(x, y, s=2, c="green")
        
        plan = self.planner.plan_buffer.readFromRT()
        if plan is not None:
            plt.scatter(plan.nominal_x[0], plan.nominal_x[1], s=2, c="blue")


if __name__ == '__main__':
    listener = PoseSubscriber()

    plt.figure(figsize=(6, 6))
    while not rospy.is_shutdown():
        plt.clf()

        listener.plot_pose()
        plt.xlim((-3, 4))
        plt.ylim((-1, 6))
        plt.pause(0.2) # display active figure and pause
    
    # rospy.spin()
