#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from threading import Lock
import csv, yaml

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

from iLQR import Track
from MPC import MPC
from cost_acc import CostACC


MAX_LIST_SIZE = 200

class PoseSubscriber:
  '''
  This class subscribes to the ros "/zed2/zed_node/pose" topic, and 
  save the most recent 200 position [x,y] in to lists

  '''
  def __init__(self):
    self.lock = Lock()
    self.x_traj = []
    self.y_traj = []
    
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



    # create pose subscriber
    rospy.loginfo("Subscribing to {}".format(pose_host_topic))
    self.sub_pose = rospy.Subscriber(pose_host_topic, PoseStamped, self.subscribe_pose_host, queue_size=1)

    def _get_ref_traj():
      self.lock.acquire()
      ref_traj = [self.x_traj, self.y_traj]
      self.lock.release()
      return Track(center_line=np.array(ref_traj), loop=False)
    
    self.cost = CostACC(params, _get_ref_traj)
    self.planner = MPC(self.cost, params, pose_topic=pose_topic, control_topic=controller_topic)

  def subscribe_pose_host(self, poseMsg):
    self.lock.acquire()
    
    while len(self.x_traj) > MAX_LIST_SIZE - 1:
      self.x_traj.pop(0)
      self.y_traj.pop(0)
    
    self.x_traj.append(-poseMsg.pose.position.x)
    self.y_traj.append(-poseMsg.pose.position.y)

    self.lock.release()
  

  def plot_pose(self):
    self.track.plot_track()
    self.track.plot_track_center()
    
    self.lock.acquire()
    plt.scatter(self.x_traj, self.y_traj, s=2, c="green")
    self.lock.release()




if __name__ == '__main__':
  listener = PoseSubscriber()

  plt.ion()
  plt.show()
  plt.figure(figsize=(5, 5))

  while not rospy.is_shutdown():
    display.clear_output(wait = True)
    display.display(plt.gcf())
    plt.clf()

    listener.plot_pose()
    plt.xlim((-5, 6))
    plt.ylim((-3, 8))
    plt.pause(0.001)
  
  # rospy.spin()
