BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 7e-4        # learning rate of the critic
WEIGHT_DECAY = 1e-10    # L2 weight decay
NOISE = 1.0             # Noise weight [0,1] 0: no noise added, 1: full noise added