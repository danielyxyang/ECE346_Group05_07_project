#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from threading import Lock
import csv, yaml, time

import rospy
from std_msgs.msg import String
from nav_msgs.msg import Odometry

from iLQR import Track
from MPC import MPC
from cost_acc import CostACC


MAX_LIST_SIZE = 100

class PoseSubscriber:
  '''
  This class subscribes to the ros "/zed2/zed_node/pose" topic, and 
  save the most recent 200 position [x,y] in to lists

  '''
  def __init__(self):
    self.lock_host = Lock()
    
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
    self.x_traj = []
    self.y_traj = []
    self.ref_track_available = False
    self.ref_track = None

    # create pose subscriber
    rospy.loginfo("Subscribing to {}".format(pose_topic))
    self.sub_pose = rospy.Subscriber(pose_topic, Odometry, self.subscribe_pose, queue_size=1)
    rospy.loginfo("Subscribing to {}".format(pose_host_topic))
    self.sub_pose_host = rospy.Subscriber(pose_host_topic, Odometry, self.subscribe_pose_host, queue_size=1)

    def _get_ref_traj():
      return self.ref_track
    
    self.cost = CostACC(params, _get_ref_traj)
    self.planner = MPC(self.cost, params, pose_topic=pose_topic, control_topic=controller_topic)

  def subscribe_pose(self, poseMsg):
    self.x = poseMsg.pose.pose.position.x
    self.y = poseMsg.pose.pose.position.y
    self.current_pos = True

  def subscribe_pose_host(self, poseMsg):
    with self.lock_host:
      # if not self.traj_init:
      #   if self.current_pos:
      #     self.x_traj = list(np.linspace(self.x, poseMsg.pose.position.x, MAX_LIST_SIZE))
      #     self.y_traj = list(np.linspace(self.y, poseMsg.pose.position.y, MAX_LIST_SIZE))
      #     self.traj_init = True
      #     rospy.loginfo("Trajectory initialized")
      #   else:
      #     return

      while len(self.x_traj) > MAX_LIST_SIZE - 1:
        self.x_traj.pop(0)
        self.y_traj.pop(0)
      
      self.x_traj.append(poseMsg.pose.pose.position.x)
      self.y_traj.append(poseMsg.pose.pose.position.y)
      ref_traj = np.array([self.x_traj, self.y_traj])
    
    if ref_traj.shape[1] <= 1:
      return
    
    try:
      self.ref_track = Track(center_line=ref_traj, loop=False)
      if not self.ref_track_available:
        rospy.loginfo("Trajectory constructed")
        self.ref_track_available = True
        self.planner.run()
    except:
      if self.ref_track_available:
        rospy.loginfo("Trajectory cannot be constructed")
        self.ref_track_available = False
        self.planner.stop()

  def plot_pose(self):
    self.track.plot_track()
    self.track.plot_track_center()
    
    with self.lock_host:
      plt.scatter(self.x_traj, self.y_traj, s=2, c="green")


if __name__ == '__main__':
  listener = PoseSubscriber()

  plt.figure(figsize=(5, 5))
  while not rospy.is_shutdown():
    plt.clf()

    listener.plot_pose()
    plt.xlim((-5, 6))
    plt.ylim((-3, 8))
    plt.pause(0.2) # display active figure and pause
  
  # rospy.spin()
