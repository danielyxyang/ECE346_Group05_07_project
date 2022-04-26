#! /bin/sh
THIS_IP=$(hostname -I)
export ROS_IP=$THIS_IP
export ROS_MASTER_URI=http://$THIS_IP:11311
export ROS_HOSTNAME=$THIS_IP

source setup.sh