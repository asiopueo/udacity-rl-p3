from collections import deque
import random
import numpy as np
from collections import namedtuple

import Network


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
        self.actor_local = Network.network_actor()
        self.actor_target = Network.network_actor()

        self.critic_local = Network.network_critic()
        self.critic_target = Network.network_critic()


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

    def update_target_nets(self, tau=0.01):
        # Implement soft update for later:
        # get_weights()[0] -- weights
        # get weights()[1] -- bias (if existent)
        # Soft-update:
        actor_weights_local = np.array( self.actor_local.get_weights() )
        actor_weights_target = np.array( self.actor_target.get_weights() )
        self.actor_target.set_weights( tau*actor_weights_local + (1-tau)*actor_weights_target )

        critic_weights_local = np.array( self.critic_local.get_weights() )
        critic_weights_target = np.array( self.critic_target.get_weights() )
        self.critic_target.set_weights( tau*critic_weights_local + (1-tau)*critic_weights_target )

    # Take action according to epsilon-greedy-policy:
    def action(self, state, epsilon=0.9):
        # Dummy action
        action_size = 2
        action = 2 * np.random.random_sample(action_size) - 1.0
        return action

    def random_action(self):
        pass


    def load_weights(self, path):
        filepath = os.path.join(path, "actor_weights_latest.ckpt")
        print("Loading actor network weights from", filepath)
        self.actor_local_net.load_weights(filepath)
        self.actor_target_net.load_weights(filepath)
        filepath = os.path.join(path, "critic_weights_latest.ckpt")
        print("Loading critic network weights from", filepath)
        self.critic_local_net.load_weights(filepath)
        self.critic_target_net.load_weights(filepath)

    def save_weights(self, path):
        filepath = os.path.join(path, "actor_weights_latest.ckpt")
        print("Saving actor network weights to", filepath)
        self.target_net.save_weights(filepath)
        filepath = os.path.join(path, "critic_weights_latest.ckpt")
        print("Saving critic network weights to", filepath)
        self.target_net.save_weights(filepath)  

 


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

