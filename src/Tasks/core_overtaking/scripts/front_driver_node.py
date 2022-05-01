#!/usr/bin/env python
import csv, yaml
import numpy as np
import matplotlib.pyplot as plt

import rospy
from nav_msgs.msg import Odometry

from MPC import MPC
from iLQR import Track
from cost_trajectory import CostTrajectory


class FrontDriver():
    def __init__(self):
        rospy.init_node('front_driver_node')
        rospy.loginfo("Start front driver node")
        
        # read parameters
        controller_topic = rospy.get_param("~ControllerTopic")
        odom_topic = rospy.get_param("~PoseTopic")
        odom_host_topic = rospy.get_param("~PoseHostTopic")
        params_file = rospy.get_param("~PlanParamsFile")
        track_file = rospy.get_param("~TrackFile")    

        # load parameters
        with open(params_file) as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
        
        # load track file
        x, y = [], []
        with open(track_file, newline='') as f:
            spamreader = csv.reader(f, delimiter=',')
            for i, row in enumerate(spamreader):
                if i > 0:
                    x.append(float(row[0]))
                    y.append(float(row[1]))
        self.track = Track(center_line=np.array([x, y]), width_left=params['track_width_L'], width_right=params['track_width_R'], loop=True)
        
        # subscribe to odometry topic
        rospy.loginfo("Subscribing to {}".format(odom_topic))
        self.sub_odom = rospy.Subscriber(odom_topic, Odometry, self.subscribe_odom, queue_size=1)
        rospy.loginfo("Subscribing to {}".format(odom_host_topic))
        self.sub_odom_host = rospy.Subscriber(odom_host_topic, Odometry, self.subscribe_odom_host, queue_size=1)
        
        self.last_p = None
        self.last_p_host = None

        # define planner
        cost = CostTrajectory(params, self.track)
        self.planner = MPC(cost, params, pose_topic=odom_topic, control_topic=controller_topic)

    def subscribe_odom(self, odom_msg):
        self.last_p = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y]

    def subscribe_odom_host(self, odom_msg):
        self.last_p_host = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y]

    def plot_pose(self):
        # plot outer loop of track
        self.track.plot_track()
        self.track.plot_track_center()

        # plot current position of front car
        if self.last_p is not None:
            plt.scatter([self.last_p[0]], [self.last_p[1]], marker="*", s=50, c="green")
        # plot current position of back car
        if self.last_p_host is not None:
            plt.scatter([self.last_p_host[0]], [self.last_p_host[1]], marker="*", s=50, c="orange")
        
        # plot iLQR trajectory of back car
        plan = self.planner.plan_buffer.readFromRT()
        if plan is not None:
            plt.scatter(plan.nominal_x[0], plan.nominal_x[1], marker="x", s=25, c="orange")


if __name__ == '__main__':
    front_driver = FrontDriver()

    plt.figure(figsize=(6, 6))
    while not rospy.is_shutdown():
        plt.clf()

        front_driver.plot_pose()
        plt.xlim((-3, 4))
        plt.ylim((-1, 6))
        plt.pause(0.2) # display active figure and pause
    
    # rospy.spin()
