# ECE346 Final Project

## Core task: Adaptive Cruise Control

### ROS Launch files
Launching host NX5
```shell
$ source setup_host.sh
$ roslaunch core_acc zed.launch NameSpace:=nx5
```

Launching client NX7
```shell
$ source setup_client.sh 05
$ roslaunch core_acc acc_planning.launch NameSpace:=nx7 NameSpaceHost:=nx5
```

## Stretch task: Platooning

### ROS Launch files
Launching host NX5
```shell
$ source setup_host.sh
$ roslaunch core_acc zed.launch NameSpace:=nx5
```

Launching client NX7 (first car)
```shell
$ source setup_client.sh 05
$ roslaunch core_acc acc_planning.launch NameSpace:=nx7 NameSpaceHost:=nx5
```

Launching client NX11 (second car)
```shell
$ source setup_client.sh 05
$ roslaunch core_acc acc_planning.launch NameSpace:=nx11 NameSpaceHost:=nx7
```