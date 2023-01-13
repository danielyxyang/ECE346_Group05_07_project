# ECE346 Final Project

## Abstract
In this project, we implemented an iLQR-based algorithm that allowed the self-driving trucks to complete adaptive cruise control and overtaking tasks. This was done by adapting the cost functions provided in previous labs and interpolating reference trajectories and velocities to ensure higher efficacy and robustness of the algorithm. These iLQR adaptations were tested by implementing them on the autonomous trucks in the MiniCity environment. The algorithm adaptations showed excellent performance on these tasks, as was seen during the in-class demonstration. This work is described here as well as discussions on shortcomings of the implementations and future experiments that could further boost the performance of autonomous navigation.

See [full report](report.pdf) for further details.

## Setup

### Installation
Clone the Git repository and setup the environment.
```shell
$ git clone https://github.com/dyang1234/ECE346_Group05_07_project.git
$ cd ECE346_Group05_07_project
$ ./setup_env.sh
```

### Set up shell environment
To set up the shell environment, source ONE of the setup files
```shell
$ source setup.sh            # without ROS network communication
```
```shell
$ source setup_client.sh 07  # set truck as client with host NX7
```
```shell
$ source setup_host.sh       # set truck as host
```

-------------------------------------------------

# ECE346 Final Project Resource

## ROS Network
ROS is a distributed computing environment. A running ROS system can comprise dozens, even hundreds of nodes, spread across multiple machines. Depending on how the system is configured, any node may need to communicate with any other node, at any time.

When running ROS over network, there should be **exactly one host** and can have multiple clients. The ROS Core is running on the host, and clients shared their messages with hosts and other clients. 

To setup ROS network, we provided simple scripts. You only need to modify the ```host.sh``` and ```client.sh``` files to replace proper IP addresses of your host and client robots. Then for each new termainal, you opened you should ```source``` the ```.sh``` file before you ```source devel/setup.bash```.

## Starting Code
We provide perception, control, and base planning codes for you to start working on Final Projects. Both control and perception codes are same as you had in the lab and self-contained, so that you will not need to do any updates with submodules. 

### Changes in planning
The new planning code still based on the iLQR that you had in labs. Instead of using a secondary PID controller to track planned trajectory, we use the planned iLQR policy directly for planning. 

### Namespace for ROS Nodes
In order to run multiple robots over network using one ROS Core, you should setup the namespace in [```src/Planning/traj_planning_ros/launch/ilqr_planning.launch```](https://github.com/SafeRoboticsLab/ECE346/blob/FinalProject/src/Planning/traj_planning_ros/launch/ilqr_planning.launch) file. Otherwise, multiple robots will publish conflicted messages using the same name. 

### If my car does not go straight 
You should adjust this line in the [```src/Control/maestro_control_ros/rc_control/cfg/calibration.cfg```](https://github.com/SafeRoboticsLab/ECE346/blob/FinalProject/src/Control/maestro_control_ros/rc_control/cfg/calibration.cfg)

```gen.add("steering_C", double_t, 0, "Calibrate the netural position of steering.", Choose a value between -1 and 1,  -1, 1)```
