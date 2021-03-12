import numpy as np
from collections import namedtuple, deque
import random
from networks_torch import Actor, Critic


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


EPISODES_BEFORE_TRAINING = 3


class MultiAgent():
    """
        Multi-Agent DDPG according to
    """
    def __init__(self, state_size, action_size, buffer_size, batch_size, gamma):
        self.agents = []
        self.agents.append( Agent(state_size=state_size, action_size=action_size, buffer_size=buffer_size, batch_size=batch_size, gamma=gamma) )
        self.agents.append( Agent(state_size=state_size, action_size=action_size, buffer_size=buffer_size, batch_size=batch_size, gamma=gamma) )
        self.num_agents = len(self.agents)

    def action(self, state, add_noise=True):
        actions = []
        for agent_id, agent in enumerate(self.agents):
            action = agent.act(np.reshape(full_states[agent_id,:], newshape=(1,-1)), i_episode, add_noise)
            action = np.reshape(action, newshape=(1,-1))            
            actions.append(action)

        actions = np.concatenate(actions, axis=0)
        return actions

    def random_action(self):
        action_size = 4
        action = 2 * np.random.random_sample(action_size) - 1.0
        return action

    def learn(self, samples, agent_no, gamma):
        #for learning MADDPG
        full_states, states, actions, rewards, full_next_states, next_states, dones = samples
        
        critic_full_next_actions = torch.zeros(states.shape[:2] + (self.action_size,), dtype=torch.float, device=DEVICE)
        for agent_id, agent in enumerate(self.maddpg_agents):
            agent_next_state = next_states[:,agent_id,:]
            critic_full_next_actions[:,agent_id,:] = agent.actor_target.forward(agent_next_state)
        critic_full_next_actions = critic_full_next_actions.view(-1, self.whole_action_dim)
        
        agent = self.maddpg_agents[agent_no]
        agent_state = states[:,agent_no,:]
        actor_full_actions = actions.clone() #create a deep copy
        actor_full_actions[:,agent_no,:] = agent.actor_local.forward(agent_state)
        actor_full_actions = actor_full_actions.view(-1, self.whole_action_dim)
                
        full_actions = actions.view(-1,self.whole_action_dim)
        
        agent_rewards = rewards[:,agent_no].view(-1,1) #gives wrong result without doing this
        agent_dones = dones[:,agent_no].view(-1,1) #gives wrong result without doing this
        experiences = (full_states, actor_full_actions, full_actions, agent_rewards, \
                       agent_dones, full_next_states, critic_full_next_actions)
        agent.learn(experiences, gamma)
        
    def update_target_nets(self):
        for agent in self.agents:
            agent.update_target_nets()

    def load_weights(self, path):
        for id, agent in enumerate(self.agents):
            filepath = os.path.join(path, "actor_weights_latest_" + str(id) + ".pth")
            print("Loading actor network weights from", filepath)
            agent.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))

            filepath = os.path.join(path, "critic_weights_latest_" + str(id) + ".pth")
            print("Loading critic network weights from", filepath)
            agent.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))
            
            self.hard_update_target_nets()

    def save_weights(self, path):
        for id, agent in enumerate(self.agents):
            filepath = os.path.join(path, "actor_weights_latest_" + str(id) + ".pth")
            print("Saving actor network weights to", filepath)
            torch.save(agent.actor_net.state_dict(), filepath) 
            filepath = os.path.join(path, "critic_weights_latest_" + str(id) + ".pth")
            print("Saving critic network weights to", filepath)
            torch.save(agent.critic_net.state_dict(), filepath)
    
    def reset():
        for agent in agents:
            agent.reset()


class Agent():
    """
        DDPG-Agent according to 
    """
    def __init__(self, state_size, action_size, buffer_size, batch_size, gamma):
        if not batch_size < buffer_size:
            raise Exception()

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        # Seed the random number generator
        random.seed()
        # QNetwork - We choose the simple network
        self.actor_local = Actor(state_size, action_size)
        self.actor_target = Actor(state_size, action_size)

        self.critic_local = Critic(state_size, action_size)
        self.critic_target = Critic(state_size, action_size)

        self.hard_update_nets()

    # Let the agent learn from experience
    def learn(self):
        # If buffer is sufficiently full, let the agent learn from his experience:
        if not agent.replay_buffer.buffer_usage():
            return

        # Retrieve batch of experiences from the replay buffer:
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample_from_buffer()

        # Batches need to be prepared before learning
        """
        for index in range(number_agents):    
            # Calculate the next q-value according to SARSA-MAX   
            # Q_new w.r.t. action:
            if not done_batch[index]:
                Q_target_batch = reward_batch + self.gamma * Q_next_state_batch
            else:
                Q_target_batch = reward_batch
        """

        full_states, actor_full_actions, full_actions, agent_rewards, agent_dones, full_next_states, critic_full_next_actions = experiences
        
        # ---------------------------- update critic ---------------------------- #
        # Get Q values from target models
        Q_target_next = self.critic_target(full_next_states, critic_full_next_actions)
        # Compute Q targets for current states (y_i)
        Q_target = agent_rewards + gamma * Q_target_next * (1 - agent_dones)
        # Compute critic loss
        Q_expected = self.critic_local(full_states, full_actions)
        critic_loss = F.mse_loss(input=Q_expected, target=Q_target) #target=Q_targets.detach() #not necessary to detach
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1.0) #clip the gradient for the critic network (Udacity hint)
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actor_loss = -self.critic_local.forward(full_states, actor_full_actions).mean() #-ve b'cse we want to do gradient ascent
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step() 


    def hard_update_nets(self):
        self.soft_update_target_nets(tau=1.0)


    # Take action according to epsilon-greedy-policy:
    def action(self, state, add_noise=True):
        if i_episode > EPISODES_BEFORE_TRAINING and self.noise_scale > NOISE_END:
            #self.noise_scale *= NOISE_REDUCTION
            self.noise_scale = NOISE_REDUCTION**(i_episode-EPISODES_BEFORE_TRAINING)

        if not add_noise:
            self.noise_scale = 0.0
                                    
        states = torch.from_numpy(states).float().to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        
        # Add noise
        actions += self.noise_scale*self.add_noise2() #works much better than OU Noise process
        #actions += self.noise_scale*self.noise.sample()
        
        return np.clip(actions, -1, 1)
            
        return action

    def random_action(self):
        action_size = 2
        action = 2 * np.random.random_sample(action_size) - 1.0
        return action

    # Copy weights from short-term model to long-term model
    def soft_update_target_nets(self, tau=0.001):
        for t, l in zip(self.actor_target.parameters(), self.actor_local.parameters() ):
            t.data.copy_( (1-tau)*t.data + tau*l.data )

        for t, l in zip(self.critic_target.parameters(), self.critic_local.parameters() ):
            t.data.copy_( (1-tau)*t.data + tau*l.data )

    def reset():
        self.noise.reset()
 


class ReplayBuffer():
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=self.buffer_size)

    # Insert experience into memory
    def insert_into_buffer(self, experience):
        self.replay_buffer.append(experience)

    # Randomly sample memory
    def sample_from_buffer(self):
        # Sample experience batch from experience buffer
        batch = random.sample(self.replay_buffer, self.batch_size)

        # Reorder experience batch such that we have a batch of states, a batch of actions, a batch of rewards, etc.
        # Eventually add 'if exp is not None'
        full_states = torch.from_numpy( np.vstack( [exp.full_states for exp in batch if exp is not None] )).float().to(device)
        states = torch.from_numpy( np.vstack( [exp.states for exp in batch if exp is not None] )).float().to(device)
        actions = torch.form_numpy( np.vstack( [exp.actions for exp in batch if exp is not None] )).float().to(device)
        rewards = torch.from_numpy( np.vstack( [exp.rewards for exp in batch if exp is not None] )).float().to(device)
        full_next_states = torch.from_numpy( np.vstack( [exp.next_states for exp in batch is not None] )).float().to(device)
        next_states = torch.from_numpy( np.vstack( [exp.full_next_states for exp in batch is not None] )).float().to(device)
        dones = torch.from_numpy( np.vstack( [exp.done for exp in batch is not None] ).astype(np.uint8)).float().to(device)

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

