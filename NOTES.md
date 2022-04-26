# General Notes

*Notes taken from https://princeton.instructure.com/groups/6959/pages/general-notes*

## Truck
- Power bank: activating pass through (i.e. charging power bank & working with Jetson)
    - Plug charging cable into DC-IN
    - Press power button 2 seconds
    - Change to 16V
    - Plug cable to Jetson into DC-OUT

## Working with the shell
- Setup shell on the Truck
    - SSH login: `ssh nx7@192.168.0.107` with password `nvidia`
    - Setup shell environment: `source ~/ece346.zsh` (or manually the commands listed next)
        ```bash
        source devel/setup.zsh # source ROS code (on zsh)
        source devel/setup.bash # source ROS code (on bash)
        
        source ~/Documents/lab2_env/bin/activate # source Python venv
        ```
    - Update submodule (if necessary): `git submodule update --init --recursive`
- Running ROS process in background (useful for inspecting ROS nodes while running them)
    - Press `CTRL+Z` to suspend process
    - Enter `bg` to run suspended process in background
    - Enter `fg` to move running process back to foreground
    - Enter `jobs` to monitor list of background processes
- Monitor resources
    - Monitor process resources: `top`
    - Monitor Jetson usage: `jtop`

## Working with ROS
- Useful ROS commands
    - ROS topics
        - List active topics: `rostopic list`
        - Show data type of topic: `rostopic type /TOPIC`
        - Show multiple messages of topic: `rostopic echo -c /TOPIC`
        - Show single message of topic: `rostopic echo -n 1 /TOPIC`
        - Show info of topic: `rostopic delay/hz/bw/info /TOPIC`
    - ROS messages
        - List messages: `rosmsg list`
        - how message structure: `rosmsg show MSG_NAME`

## GitHub
- Cloning private group repository (instead of public class repository)
    ```
    git clone https://github.com/dyang1234/ECE346_Group07.git
    git submodule update --init --recursive
    git remote add upstream https://github.com/SafeRoboticsLab/ECE346.git
    git remote set-url --push upstream DISABLE
    ```
- SSH key (required for `git submodule update --init --recursive`)
    - Generate SSH key: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
    - Add SSH key to your GitHub account: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account
- Accessing repository over command line (from the truck)
    - Username: `dyang1234`
    - Password (access token): `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`