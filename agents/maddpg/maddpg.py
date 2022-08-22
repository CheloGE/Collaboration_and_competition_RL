import torch
from agents.maddpg.ddpg.ddpg_agent import DDPG_Agent
from agents.maddpg import parameters
from agents.maddpg.ddpg.helpers import ReplayBuffer
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using device: {device}')

## This algorithm is adapted from the maddpg version in the RL Udacity lessons
class MADDPG:
    def __init__(self, no_agents, state_size, action_size, random_seed):
        
        self.no_agents = no_agents
        self.state_size = state_size
        
        # Creating number of independent agents 
        self.maddpg_agents = [DDPG_Agent(no_agents, state_size, action_size, random_seed) for _ in range(no_agents)]
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, parameters.BUFFER_SIZE, parameters.BATCH_SIZE, random_seed)
        

    def act(self, states_all_agents, noise=parameters.NOISE):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(state, noise) for agent, state in zip(self.maddpg_agents, states_all_agents)]
        return np.array(actions).reshape(1,-1) # Flatten actions of all agents in a single vector

    
    def step(self, agents_state, agents_actions, agents_rewards, agents_next_state, agents_dones):
        states = agents_state.reshape(1,-1) # flatten the compound states from (no_agents, state_size) to (1, no_agents * state_size)
        next_states = agents_next_state.reshape(1,-1) # flatten next states, same way as states
        self.memory.add(states, agents_actions, agents_rewards, next_states, agents_dones)

        # Learn, if enough samples are available in memory
        if len(self.memory) > parameters.BATCH_SIZE:
            experiences_batch_for_all_agents = [self.memory.sample() for _ in range(self.no_agents)]
            self.learn(experiences_batch_for_all_agents, parameters.GAMMA)

    def learn(self, experiences_batch_for_all_agents, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Params
        ======
            experiences_batch_for_all_agents (list[torch.Tensor]): list of  [batch, no_agents, (s, a, r, s', done)]
            gamma (float): discount factor
        """

        
        
        next_actions_batch_all_agents = []
        # Get next actions
        for i, agent in enumerate(self.maddpg_agents):
            
            agent_ind = torch.tensor([i]).to(device) # agent index
            _, _, _, next_state_batch, _ = experiences_batch_for_all_agents[i] # extracting experience info for each agent

            # reshape back to non flatten data to extract the info for agent i
            next_state_batch = next_state_batch.reshape(-1, self.no_agents, self.state_size) 
            next_state_batch = next_state_batch.index_select(dim=1, index=agent_ind).squeeze(1)
            # next actions batch based on next states batch
            next_action_batch = agent.actor_target(next_state_batch)
            next_actions_batch_all_agents.append(next_action_batch)
        
        # each agent learns from its experience sample
        for i, ddpg_agent in enumerate(self.maddpg_agents):
            ddpg_agent.learn(i, experiences_batch_for_all_agents[i], gamma, next_actions_batch_all_agents)



    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            # ----------------------- update target networks ----------------------- #
            ddpg_agent.soft_update(ddpg_agent.critic_local, ddpg_agent.critic_target, parameters.TAU)
            ddpg_agent.soft_update(ddpg_agent.actor_local, ddpg_agent.actor_target, parameters.TAU)      
            