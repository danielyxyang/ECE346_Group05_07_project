# ECE346 Final Project

## Core task: Overtaking

### ROS Launch files
Launching host NX5
```shell
$ source setup_host.sh
$ roslaunch core_overtaking front_driver.launch NameSpace:=nx5
```

Launching client NX7
```shell
$ source setup_client.sh 05
$ roslaunch core_overtaking overtaker.launch NameSpace:=nx7 NameSpaceHost:=nx5
```
