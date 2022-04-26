#! /bin/sh
HOST_IP="192.168.0.1${1:-07}" # default host: NX7
THIS_IP=$(hostname -I | grep -o "192\.168\.0\.1..")
export ROS_IP=$THIS_IP
export ROS_MASTER_URI=http://$HOST_IP:11311
export ROS_HOSTNAME=$THIS_IP

source setup.sh