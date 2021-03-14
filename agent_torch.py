from collections import namedtuple, deque
import random
import os, copy
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from networks_torch import Actor, Critic


# Define named tuple 'Experience'; you can use a dictionary alternatively
Experience = namedtuple('Experience', ['full_state', 'state', 'action', 'reward', 'full_next_state', 'next_state', 'done'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPISODES_BEFORE_TRAINING = 3
NOISE_START = 1.0
NOISE_START=1.0
NOISE_END=0.1
NOISE_REDUCTION=0.999
EPISODES_BEFORE_TRAINING = 300
NUM_LEARN_STEPS_PER_ENV_STEP = 3


class MultiAgent():
    """
        Multi-Agent DDPG according to
    """
    def __init__(self, state_size, action_size, buffer_size, batch_size, gamma, learn_rate):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        random.seed()
        np.random.seed()

        self.agents = []
        self.agents.append( Agent(state_size=state_size, action_size=action_size, buffer_size=buffer_size, batch_size=batch_size, gamma=gamma, learn_rate=learn_rate) )
        self.agents.append( Agent(state_size=state_size, action_size=action_size, buffer_size=buffer_size, batch_size=batch_size, gamma=gamma, learn_rate=learn_rate) )
        self.num_agents = len(self.agents)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)

    def action(self, states, episode, add_noise=True):
        actions = []
        for agent_id, agent in enumerate(self.agents):
            #action = agent.action(states[agent_id,:], episode, add_noise) # Wrong shape before stacking the individual actions
            action = agent.action(np.reshape(states[agent_id,:], newshape=(1,-1)), episode, add_noise)
            action = np.reshape(action, newshape=(1,-1))            
            actions.append(action)

        actions = np.concatenate(actions, axis=0)
        return actions

    def random_action(self):
        action_size = 4
        action = 2 * np.random.random_sample(action_size) - 1.0
        return action

    # Prepares batches before actual learning is done by the agents
    def learn(self):
        # If buffer is sufficiently full, let the agent learn from his experience:
        if not self.replay_buffer.buffer_usage():
            return

        full_states, states, actions, rewards, full_next_states, next_states, dones = self.replay_buffer.sample_from_buffer()
                 
        # states.shape[:2] = (64,2)
        # states.shape[:2] + (self.action_size,) = (64,2,2)
        critic_full_next_actions = torch.zeros(states.shape[:2] + (self.action_size,), dtype=torch.float, device=device)

        for agent_idx, agent in enumerate(self.agents):
            agent_next_state = next_states[:,agent_idx,:]
            critic_full_next_actions[:, agent_idx,:] = agent.actor_target.forward(agent_next_state)
            
            agent_state = states[:, agent_idx,:]
            actor_full_actions = actions.clone() # deep copy
            actor_full_actions[:, agent_idx,:] = agent.actor_local.forward(agent_state)
            actor_full_actions = actor_full_actions.view(-1, self.action_size * self.num_agents)
            
            agent_rewards = rewards[:,agent_idx].view(-1,1) # Wrong result without this
            agent_dones = dones[:,agent_idx].view(-1,1)     # Wrong result without this

        critic_full_next_actions = critic_full_next_actions.view(-1, self.action_size * self.num_agents)                  
        full_actions = actions.view(64, self.action_size * self.num_agents)
        

        agent_exp = (full_states, full_actions, agent_rewards, full_next_states,  agent_dones, actor_full_actions, critic_full_next_actions)
        
        agent.learn( agent_exp )
        


    def update_target_nets(self):
        for agent in self.agents:
            agent.update_target_nets()

    def load_weights(self, path):
        for id, agent in enumerate(self.agents):
            filepath = os.path.join(path, "actor_weights_latest_" + str(id) + ".pth")
            print("Loading actor network weights from", filepath)
            agent.actor_local.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))

            filepath = os.path.join(path, "critic_weights_latest_" + str(id) + ".pth")
            print("Loading critic network weights from", filepath)
            agent.critic_local.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))
            
            self.hard_update_target_nets()

    def save_weights(self, path):
        for id, agent in enumerate(self.agents):
            filepath = os.path.join(path, "actor_weights_latest_" + str(id) + ".pth")
            print("Saving actor network weights to", filepath)
            torch.save(agent.actor_local.state_dict(), filepath) 
            filepath = os.path.join(path, "critic_weights_latest_" + str(id) + ".pth")
            print("Saving critic network weights to", filepath)
            torch.save(agent.critic_local.state_dict(), filepath)
    
    def reset(self):
        for agent in self.agents:
            agent.reset()


class Agent():
    """
        DDPG-Agent according to 
    """
    def __init__(self, state_size, action_size, buffer_size, batch_size, gamma, learn_rate):
        if not batch_size < buffer_size:
            raise Exception()

        self.state_size = state_size
        self.action_size = action_size

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.gamma = gamma

        self.noise = OUNoise(action_size)
        self.noise_scale = NOISE_START

        # QNetwork - We choose the simple network
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)

        self.critic_local = Critic(2*state_size, 2*action_size).to(device)
        self.critic_target = Critic(2*state_size, 2*action_size).to(device)

        self.actor_optimizer = optim.Adam( self.actor_local.parameters(), lr=self.learn_rate )
        self.critic_optimizer = optim.Adam( self.critic_local.parameters(), lr=self.learn_rate )

        self.hard_update_nets()


    # Take action according to epsilon-greedy-policy:
    def action(self, state, i_episode, add_noise=True):
        if i_episode > EPISODES_BEFORE_TRAINING and self.noise_scale > NOISE_END:
            #self.noise_scale *= NOISE_REDUCTION
            self.noise_scale = NOISE_REDUCTION**(i_episode-EPISODES_BEFORE_TRAINING)

        if not add_noise:
            self.noise_scale = 0.0
                                    
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()


        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        # Add noise
        #actions += self.noise_scale*self.noise.sample()
        action += self.noise_scale*self.add_noise2() #works much better than OU Noise process
        return np.clip(action, -1, 1)


    def random_action(self):
        action_size = 2
        action = 2 * np.random.random_sample(action_size) - 1.0
        return action


    # Let the agent learn from experience
    def learn(self, experiences):
        full_states, full_actions, agent_rewards, full_next_states, agent_dones, actor_full_actions, critic_full_next_actions = experiences
        
        # Get Q values from target models
        Q_target_next = self.critic_target(full_next_states, critic_full_next_actions)
        # Compute Q targets for current states (y_i)
        Q_target = agent_rewards + self.gamma * Q_target_next * (1 - agent_dones)
        # Compute critic loss
        Q_expected = self.critic_local(full_states, full_actions)
        critic_loss = F.mse_loss(input=Q_expected, target=Q_target) #target=Q_targets.detach() #not necessary to detach
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1.0) #clip the gradient for the critic network (Udacity hint)
        self.critic_optimizer.step()
        
        actor_loss = -self.critic_local.forward(full_states, actor_full_actions).mean() #-ve b'cse we want to do gradient ascent
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step() 


    # Copy weights from short-term model to long-term model
    def soft_update_target_nets(self, tau=0.001):
        for t, l in zip(self.actor_target.parameters(), self.actor_local.parameters() ):
            t.data.copy_( (1-tau)*t.data + tau*l.data )

        for t, l in zip(self.critic_target.parameters(), self.critic_local.parameters() ):
            t.data.copy_( (1-tau)*t.data + tau*l.data )

    def hard_update_nets(self):
        self.soft_update_target_nets(tau=1.0)


    def add_noise2(self):
        noise = 0.5*np.random.randn(1, self.action_size) #sigma of 0.5 as sigma of 1 will have alot of actions just clipped
        return noise

    def reset(self):
        self.noise.reset()
 


class ReplayBuffer():
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=self.buffer_size)

    # Insert experience into memory
    def insert_into_buffer(self, state, action, reward, next_state, done):        
        full_state = state.flatten()
        full_next_state = next_state.flatten()
        #full_state = np.reshape(state, newshape=(1,2,-1))
        #full_next_state = np.reshape(next_state, newshape=(1,2,-1))

        exp = Experience(full_state, state, action, reward, full_next_state, next_state, done)
        self.replay_buffer.append(exp)

    # Randomly sample memory
    def sample_from_buffer(self):
        # Sample experience batch from experience buffer
        batch = random.sample(self.replay_buffer, self.batch_size)

        # Reorder experience batch such that we have a batch of states, a batch of actions, a batch of rewards, etc.
        # Eventually add 'if exp is not None'
        full_states = torch.from_numpy( np.array( [exp.full_state for exp in batch if exp is not None] )).float().to(device)
        states = torch.from_numpy( np.array( [exp.state for exp in batch if exp is not None] )).float().to(device)
        actions = torch.from_numpy( np.array( [exp.action for exp in batch if exp is not None] )).float().to(device)
        rewards = torch.from_numpy( np.array( [exp.reward for exp in batch if exp is not None] )).float().to(device)
        full_next_states = torch.from_numpy( np.array( [exp.full_next_state for exp in batch if exp is not None] )).float().to(device)
        next_states = torch.from_numpy( np.array( [exp.next_state for exp in batch if exp is not None] )).float().to(device)
        dones = torch.from_numpy( np.array( [exp.done for exp in batch if exp is not None] ).astype(np.uint8)).float().to(device)

        return full_states, states, actions, rewards, full_next_states, next_states, dones

    # Get length of memory
    def buffer_usage(self):
        return len(self.replay_buffer) > self.batch_size



class Noise():
    def __init__(self):
        pass

    def sample(self):
        mu = 0.0
        sigma = 1.0
        return np.random.random(mu, sigma)


# Ornstein-Uhlenbeck process:
class OUNoise():
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu + np.ones(size)
        self.sigma = sigma
        self.theta = theta
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state += dx
        return self.state
