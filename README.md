# Continuous Control with Deep Deterministic Policy Gradient

In this project, I trained an double-jointed arm to move to target locations.

## Reinforcement Learning Environment

Unity Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. The gif below shows the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment for this project.

![](reacher.gif)

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. The agents must get an average score of +30 (over 100 consecutive episodes, and over all agents) to solve the environment (version 1 or 2).

## Getting Started

1. Create the Conda Environment

    a. Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer, by selecting the latest Python version for your operating system. If you already have `conda` or `miniconda` installed, you should be able to skip this step and move on to step b.

    **Download** the latest version of `miniconda` that matches your system.

    |        | Linux | Mac | Windows |
    |--------|-------|-----|---------|
    | 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
    | 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

    [win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
    [win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
    [mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    [lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    [lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

    **Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

    - **Linux:** http://conda.pydata.org/docs/install/quick.html#linux-miniconda-install
    - **Mac:** http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install
    - **Windows:** http://conda.pydata.org/docs/install/quick.html#windows-miniconda-install

    b. Install git and clone the repository.

    For working with Github from a terminal window, you can download git with the command:
    ```
    conda install git
    ```
    To clone the repository, run the following command:
    ```
    cd PATH_OF_DIRECTORY
    git clone https://github.com/iDataist/Navigation-with-Deep-Q-Network
    ```
    c. Create local environment

    - Create (and activate) a new environment, named `ddpg-env` with Python 3.7. If prompted to proceed with the install `(Proceed [y]/n)` type y.

        - __Linux__ or __Mac__:
        ```
        conda create -n ddpg-env python=3.7
        conda activate ddpg-env
        ```
        - __Windows__:
        ```
        conda create --name ddpg-env python=3.7
        conda activate ddpg-env
        ```

        At this point your command line should look something like: `(ddpg-env) <User>:USER_DIR <user>$`. The `(ddpg-env)` indicates that your environment has been activated, and you can proceed with further package installations.

    - Install a few required pip packages, which are specified in the requirements text file. Be sure to run the command from the project root directory since the requirements.txt file is there.
        ```
        pip install -r requirements.txt
        ipython3 kernel install --name ddpg-env --user
        ```
    - Open Jupyter Notebook, and open the Continuous_Control.ipynb file. Run all the cells in the jupyter notebook to train the agents.
        ```
        jupyter notebook
        ```
2. Download the Unity Environment

   a. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

   - Version 1: One (1) Agent
       - Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
       - Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
       - Windows (32-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
       - Windows (64-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
   - Version 2: Twenty (20) Agents
       - Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
       - Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
       - Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
       - Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

    (For Windows users) Check out this [link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or this [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

    b. Place the file in the folder with the jupyter notebook, and unzip (or decompress) the file.

## File Descriptions

1. [requirements.txt](https://github.com/iDataist/Continuous-Control-with-Deep-Deterministic-Policy-Gradient/blob/main/requirements.txt) - Includes all the required libraries for the Conda Environment.
2. [model.py](https://github.com/iDataist/Continuous-Control-with-Deep-Deterministic-Policy-Gradient/blob/main/model.py) - Defines the actor and critic networks.
3. [ddpg_agent.py](https://github.com/iDataist/Continuous-Control-with-Deep-Deterministic-Policy-Gradient/blob/main/ddpg_agent.py) -  Defines the Agent that uses DDPG to determine the best action to take and maximizes the overall or total reward.
4. [Continuous_Control.ipynb](https://github.com/iDataist/Continuous-Control-with-Deep-Deterministic-Policy-Gradient/blob/main/Continuous_Control.ipynb) - The main file that trains the actor and critic networks. This file can be run in the Conda environment.

