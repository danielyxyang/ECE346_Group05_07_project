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

class PoseSubscriber:
    def __init__(self):
        rospy.init_node('acc_planning_node')
        rospy.loginfo("Start ACC planning node")

        # read arguments
        controller_topic = rospy.get_param("~ControllerTopic")
        odom_topic = rospy.get_param("~PoseTopic")
        odom_host_topic = rospy.get_param("~PoseHostTopic")
        params_file = rospy.get_param("~PlanParamsFile")
        track_file = rospy.get_param("~TrackFile")

        # load parameters
        with open(params_file) as file:
            self.params = yaml.load(file, Loader=yaml.FullLoader)
        
        self.T = self.params['T']
        self.N = self.params['N']
        self.replan_dt = self.T / (self.N - 1)

        self.max_rate_time = 0.01
        self.max_rate_space = 0.05

        self.v_max = self.params['v_max']
        self.v_min = self.params['v_min']
        self.safety_distance = self.params['safety_distance']
        
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
        self.traj_host = Trajectory(max_list_size=100)

        # create pose subscriber
        rospy.loginfo("Subscribing to {}".format(odom_topic))
        self.sub_odom = rospy.Subscriber(odom_topic, Odometry, self.subscribe_odom, queue_size=1)
        rospy.loginfo("Subscribing to {}".format(odom_host_topic))
        self.sub_odom_host = rospy.Subscriber(odom_host_topic, Odometry, self.subscribe_odom_host, queue_size=1)

        def _get_ref_traj(n=None):
            if self.traj.size() > 0:
                prefix = [self.traj.get_trajectory()[-1]]
            else:
                prefix = []
            ref_trajectory = self.traj_host.get_reference_trajectory(min_size=n, prefix=prefix)
            if n is not None:
                return ref_trajectory[:, :n]
            else:
                return ref_trajectory
        
        self.cost = CostACC(self.params, _get_ref_traj)
        self.planner = MPC(self.cost, self.params, pose_topic=odom_topic, control_topic=controller_topic)

    def subscribe_odom(self, odom_msg):
        current_t = odom_msg.header.stamp
        if self.traj.size() > 0:
            state = self.traj.get_trajectory()[-1]
            last_t = state.t
        else:
            last_t = None
        
        # limiting sampling rate over time
        if last_t is not None and (current_t - last_t).to_sec() < self.max_rate_time:
            return
        
        # add state to trajectory
        self.traj.add_odom_state(odom_msg)
        # self.traj_host.update_odom_state(odom_msg, 0)

        # truncate host trajectory up to current position
        self.traj_host.truncate_trajectory(odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y)

        # set reference velocity
        self.update_reference_velocity()

    def subscribe_odom_host(self, odom_msg):
        current_t = odom_msg.header.stamp
        current_p = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])
        if self.traj_host.size() > 0:
            state = self.traj_host.get_trajectory()[-1]
            last_t = state.t
            last_p = state.state[0:2]
        else:
            last_t = None
            last_p = None

        # limiting sampling rate over time
        if last_t is not None and (current_t - last_t).to_sec() < self.max_rate_time:
            return
        # limiting sampling rate over space
        if last_p is not None and np.linalg.norm(current_p - last_p) < self.max_rate_space:
            return
        
        # add state to trajectory
        self.traj_host.add_odom_state(odom_msg)
        
        # set reference velocity
        self.update_reference_velocity()

    def update_reference_velocity(self):
        if self.traj.size() > 0 and self.traj_host.size() > 0:
            # compute distance to host
            state = self.traj.get_trajectory()[-1]
            state_host = self.traj_host.get_trajectory()[-1]
            distance_to_host = np.linalg.norm(state_host.state[0:2] - state.state[0:2]) + self.traj_host.length() # TODO
            
            distance_diff = distance_to_host - self.safety_distance
            if distance_diff >= 0:
                target_velocity = state_host.state[2] + distance_diff # increase speed
            else:
                target_velocity = state_host.state[2] + distance_diff # reduce speed
            target_velocity = np.clip(target_velocity, self.v_min, self.v_max)
            self.traj_host.set_reference_velocity(target_velocity, self.replan_dt)

            print("host_velocity: {:.6f}, target_velocity: {:.6f}, distance_diff: {:.6f}, d: {:.6f}".format(state_host.state[2], target_velocity, distance_diff, target_velocity * self.replan_dt))


    def plot_pose(self):
        # plot outer loop of track
        self.track.plot_track()
        self.track.plot_track_center()

        # plot current position of back car
        if self.traj.size() > 0:
            last_state = self.traj.get_trajectory()[-1]
            plt.scatter([last_state.state[0]], [last_state.state[1]], c="orange", marker="*")
        # plot current position of front car
        if self.traj_host.size() > 0:
            last_state = self.traj_host.get_trajectory()[-1]
            plt.scatter([last_state.state[0]], [last_state.state[1]], c="green", marker="*")
        
        # plot reference trajectory of front car
        if self.traj.size() > 0:
            prefix = [self.traj.get_trajectory()[-1]]
        else:
            prefix = []
        states = self.traj_host.get_reference_trajectory(min_size=self.N, prefix=prefix)
        plt.scatter(states[0], states[1], s=2, c="green")
        # plot iLQR trajectory of back car
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
