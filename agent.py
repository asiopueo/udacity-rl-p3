from collections import deque
import random
import numpy as np
from collections import namedtuple

from networks_torch import Actor, Critic


class MultiAgent():
    """
        Multi-Agent DDPG according to
    """
    def __init__(self, buffer_size, batch_size, gamma):
        self.agents = []
        self.agents.append( Agent(buffer_size=buffer_size, batch_size=batch_size, gamma=gamma) )
        self.agents.append( Agent(buffer_size=buffer_size, batch_size=batch_size, gamma=gamma) )
        self.num_agents = len(self.agents)

    def action(self, state):
        action = []
        for agent in self.agents:
            action.append( agent.action(state) )

        return action


    def learn(self):
        for agent in self.agents:
            agent.learn()
        
    def update_target_nets(self):
        for agent in self.agents:
            agent.update_target_nets()
    



class Agent():
    """
        DDPG-Agent according to 
    """
    def __init__(self, buffer_size, batch_size, gamma):
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
        self.actor_local = Actor()
        self.actor_target = Actor()

        self.critic_local = Critic()
        self.critic_target = Critic()


    # Let the agent learn from experience
    def learn(self):
        # Retrieve batch of experiences from the replay buffer:
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample_from_buffer()
        # Prepare the target. Note that Q_target[:,action] will need to be assigned the 'true' learning target.
        Q_target = self.local_net.predict( state_batch )
        Q_next_state = np.max( self.target_net.predict(next_state_batch), axis=1 )


        # Batches need to be prepared before learning
        for index in range(number_agents):    
            # Calculate the next q-value according to SARSA-MAX   
            # Q_new w.r.t. action:
            if not done_batch[index]:
                Q_target_batch = reward_batch + self.gamma * Q_next_state_batch
            else:
                Q_target_batch = reward_batch

            # Update actor:
            #self.actor_local.fit(, None, batch_size=self.batch_size, epochs=1, shuffle=Flase, verbose=1)

            # Update critic:
            #self.critic_local.fit(X_np, Q_target_batch, batch_size=self.batch_size, epochs=1, shuffle=False, verbose=1)
        
        # Soft updates of target nets:
        self.update_target_nets()


    # Take action according to epsilon-greedy-policy:
    def action(self, state, epsilon=0.9):
        # Dummy action
        action_size = 2
        action = 2 * np.random.random_sample(action_size) - 1.0
        return action

    def random_action(self):
        pass

    # Copy weights from short-term model to long-term model
    def soft_update_target_nets(self, tau=0.001):
        for t, l in zip(self.actor_target.parameters(), self.actor_local.parameters() ):
            t.data.copy_( (1-tau)*t.data + tau*l.data )

        for t, l in zip(self.critic_target.parameters(), self.critic_local.parameters() ):
            t.data.copy_( (1-tau)*t.data + tau*l.data )


    def load_weights(self, path):
        filepath = os.path.join(path, "actor_weights_latest.pth")
        print("Loading actor network weights from", filepath)
        self.actor_local.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))

        filepath = os.path.join(path, "critic_weights_latest.pth")
        print("Loading critic network weights from", filepath)
        self.critic_local.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))
        
        self.hard_update_target_nets()


    def save_weights(self, path):
        filepath = os.path.join(path, "actor_weights_latest.pth")
        print("Saving actor network weights to", filepath)
        torch.save(self.actor_net.state_dict(), filepath) 
        filepath = os.path.join(path, "critic_weights_latest.pth")
        print("Saving critic network weights to", filepath)
        torch.save(self.critic_net.state_dict(), filepath) 


 


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
        state = np.vstack( [exp.state for exp in batch] )
        action = np.vstack( [exp.action for exp in batch] )
        reward = np.vstack( [exp.reward for exp in batch] )
        state_next = np.vstack( [exp.next_state for exp in batch] )
        done = np.vstack( [exp.done for exp in batch] )

        return state, action, reward, state_next, done

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

