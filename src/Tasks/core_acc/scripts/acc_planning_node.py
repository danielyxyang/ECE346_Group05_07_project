#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import csv, yaml

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
            params = yaml.load(file, Loader=yaml.FullLoader)
        
        # load track
        x = []
        y = []
        with open(track_file, newline='') as f:
            spamreader = csv.reader(f, delimiter=',')
            for i, row in enumerate(spamreader):
                if i > 0:
                    x.append(float(row[0]))
                    y.append(float(row[1]))
        self.track = Track(center_line=np.array([x, y]), width_left=params['track_width_L'], width_right=params['track_width_R'], loop=True)

        # initialize trajectory
        self.traj = Trajectory(max_list_size=1)
        self.traj_host = Trajectory(max_list_size=MAX_LIST_SIZE)
        self.last_t = None

        # create pose subscriber
        rospy.loginfo("Subscribing to {}".format(pose_topic))
        self.sub_pose = rospy.Subscriber(pose_topic, Odometry, self.subscribe_pose, queue_size=1)
        rospy.loginfo("Subscribing to {}".format(pose_host_topic))
        self.sub_pose_host = rospy.Subscriber(pose_host_topic, Odometry, self.subscribe_pose_host, queue_size=1)

        def _get_ref_traj():
            return self.traj_host.get_trajectory()
        
        self.cost = CostACC(params, _get_ref_traj)
        self.planner = MPC(self.cost, params, pose_topic=pose_topic, control_topic=controller_topic)

    def subscribe_pose(self, poseMsg):
        self.traj.add_point(poseMsg.pose.pose.position.x, poseMsg.pose.pose.position.y)

    def subscribe_pose_host(self, odomMsg):
        if self.last_t is not None and (odomMsg.header.stamp - self.last_t).to_sec() < 0.25:
            return
        
        self.traj_host.add_point(odomMsg.pose.pose.position.x, odomMsg.pose.pose.position.y)
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

            # while len(self.x_traj) > MAX_LIST_SIZE - 1:
            #     self.x_traj.pop(0)
            #     self.y_traj.pop(0)
            
            # self.x_traj.append(poseMsg.pose.pose.position.x)
            # self.y_traj.append(poseMsg.pose.pose.position.y)
            # ref_traj = np.array([self.x_traj, self.y_traj])
        
        # if ref_traj.shape[1] <= 1:
        #     return
        
        # try:
        #     self.ref_track = Track(center_line=ref_traj, loop=False)
        #     if not self.ref_track_available:
        #         rospy.loginfo("Trajectory constructed")
        #         self.ref_track_available = True
        #         self.planner.run()
        # except:
        #     if self.ref_track_available:
        #         rospy.loginfo("Trajectory cannot be constructed")
        #         self.ref_track_available = False
        #         self.planner.stop()

    def plot_pose(self):
        self.track.plot_track()
        self.track.plot_track_center()
        
        x, y, x_mid, y_mid = self.traj_host.get_trajectory()
        # plt.scatter(x_mid, y_mid, s=2, c="red")
        plt.scatter(x, y, s=2, c="green")
        plt.scatter(x[-1:], y[-1:], c="green", marker="*")
        
        plan = self.planner.plan_buffer.readFromRT()
        plt.scatter(plan.nominal_x[0], plan.nominal_x[1], s=2, c="red")
        
        x, y, _, _ = self.traj.get_trajectory()
        plt.scatter(x, y, c="orange", marker="*")


        # t = self.cost.ref_traj_midpoints.T
        # plt.scatter(t[0], t[1], s=2, c="red")


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
