# Collaboration_and_competition_RL

The goal of this repo is to train two agents in the form of tenis rackets and mantain the ball in play as longer as possible. i.e. The goal of each agent is to keep the ball in play.

## Project Details

The below illustration shows the final outcome of the current solution that uses a multi agent actor-critic mehod to solve the problem.

> TODO: Here goes the multiagent image final outcome

The observation space consists of 24 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

* Continuous state space: 24
* Continuous action space: 2
* number of agents: 2

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.

The environment is **considered solved**, when the **average (over 100 episodes)** of those scores is at least `+0.5`.

For more details on the project solution please read the [report.md](report.md) file.

## Environment setup

All commands below must be run in the project's folder `cd <path_to_root_folder_of_this_project>`

### First lets donwload all required files

* Make sure to download the environment builded in unity with ml_agents extension

    For linux:
    
    * Two-racket tennis scenario from [mlagents](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#tennis). You can find the binaries [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    

### Install dependencies with docker

For this project a docker container was created, please follow the following steps to setup the environment:

1. Build Dockerfile

    `docker build -t unity_ml_agents:pytorch .`

2. Create container (this assumes you want to run the project on GPU)

    ``docker run --name mlagents_unity -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v `pwd`:/project -it --env QT_X11_NO_MITSHM=1 --device /dev/dri --privileged --gpus all unity_ml_agents:pytorch``

3. Everytime we want to run container

    `docker start mlagents_unity`

    `docker exec -it mlagents_unity bash`

## How to run

The main script is in the [tennis.ipynb](tennis.ipynb) file, where you can run a dummy model that selects and action randomly. You can also train a MADDPG model with 2 agents and use it in inference of new cases.

> TODO: Update the below folders with this project folders

If you want to check the agents model you can find it under the folder: [agents/ddpg/](agents/ddpg/). In this folder the main logic for the agent models is implemented in the [ddpg_agent.py](agents/ddpg/ddpg_agent.py). 

* [actor_critic_models.py](agents/ddpg/actor_critic_models.py) constains the two neural networks used to build the ddpg agent
* [helpers.py](agents/ddpg/helpers.py) constains all ddpg funtions that assist ddpg logic such as the replay buffer and the Ornstein-Uhlenbeck noise.
* Finally the [parameters.py](agents/ddpg/parameters.py) script constains all the parameters that could be changed in this project. If you are interested in trying different training processes this is the file you are looking for.
