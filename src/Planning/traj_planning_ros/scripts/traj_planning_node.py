#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import numpy as np
from MPC import MPC
from iLQR import Track
from cost_trajectory import CostTrajectory
import sys, os, csv, yaml


def main():
    rospy.init_node('traj_planning_node')
    rospy.loginfo("Start trajectory planning node")
    ## read parameters
    ControllerTopic = rospy.get_param("~ControllerTopic")
    PoseTopic = rospy.get_param("~PoseTopic")
    ParamsFile = rospy.get_param("~PlanParamsFile")
    TrackFile = rospy.get_param("~TrackFile")    

    # load parameters
    with open(ParamsFile) as file:
      params = yaml.load(file, Loader=yaml.FullLoader)
    
    # load track file
    x = []
    y = []
    with open(TrackFile, newline='') as f:
        spamreader = csv.reader(f, delimiter=',')
        for i, row in enumerate(spamreader):
            if i > 0:
                x.append(float(row[0]))
                y.append(float(row[1]))

    center_line = np.array([x, y])
    track = Track(center_line=center_line,
                  width_left=params['track_width_L'],
                  width_right=params['track_width_R'],
                  loop=True)

    # define cost function
    cost = CostTrajectory(params, track)

    planner = MPC(cost, params,
                  pose_topic=PoseTopic,
                  control_topic=ControllerTopic)
    rospy.spin()


if __name__ == '__main__':
    main()
