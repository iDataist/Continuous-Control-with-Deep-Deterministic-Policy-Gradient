## Deep Q-Networks

### 1. Learning algorithm
  - #### Value-based Deep Reinforcement Learning
    - Reinforcement Learning (RL) is a branch of Machine Learning, where an agent outputs an action and the environment returns an observation (the state of the system) and a reward. The goal of an agent is to determine the best action to take and maximizes the overall or total reward.
    - Value-based Deep RL uses nonlinear function approximators (Deep Neural Network) to calculate the value actions based directly on observation from the environment. Deep Learning can be used to find the optimal parameters for these function approximators.
  - #### Experience Replay
    - I created a ReplayBuffer Class to enable experience replay<sup>1, 2</sup>. Using the replay pool, the behavior distribution is averaged over many of its previous states, smoothing out learning and avoiding oscillations. The advantage is that each step of the experience is potentially used in many weight updates.

### 2. Model architecture for the neural network
  - #### Fixed Q-Targets
    - I adopted Double Deep Q-Network structure<sup>1, 2</sup> with three fully connected layers. If a single network is used, the Q-functions values change at each step of training, and then the value estimates can quickly spiral out of control. I used a target network to represent the old Q-function, which is used to compute the loss of every action during training.
  - #### Dueling Networks
    - I adopted the dueling networks structure<sup>3</sup>. Dueling networks use two streams, one that estimates the state value function and one that estimates the advantage for each action.These streams may share some layers in the beginning, then branch off with their own fully-connected layers. The desired Q values are obtained by combining the state and advantage values. The value of most states don't vary a lot across actions. So, it makes sense to try and directly estimate them. But we still need to capture the difference actions make in each state. This is where the advantage function comes in.

### 3. Hyperparameters

  - #### Replay buffer size
    - BUFFER_SIZE = int(1e5)
  - #### Minibatch size
    - BATCH_SIZE = 64
  - #### Discount factor
    - GAMMA = 0.99
  - #### For soft update of target parameters
    - TAU = 1e-3
  - #### Learning rate
    - LR = 5e-4
  - #### How often to update the network
    - UPDATE_EVERY = 4