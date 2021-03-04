from collections import deque
import random
import numpy as np
from collections import namedtuple
import tensorflow as tf


import network


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

    def load_weights(self):
        for agent in self.agents:
            agent.load_weights()
    



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

        # Update critic:
        with tf.GradientTape as tape:
            target_actions = self.target_net(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training = True)
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y-critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients( zip(critic_grad, critic_model.trainable_variables) )


        # Update actor:
        with tf.GradientTape as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, action_batch], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients( zip(actor_grad, actor_model.trainable_variables) )

        # Soft updates of target nets:
        self.update_target_nets()

    def update_target_nets(self, tau=0.01):
        # Implement soft update for later:
        # get_weights()[0] -- weights
        # get weights()[1] -- bias (if existent)
        # Soft-update:
        actor_weights_local = np.array( self.actor_local.get_weights(), dtype=object )
        actor_weights_target = np.array( self.actor_target.get_weights(), dtype=object )
        self.actor_target.set_weights( tau*actor_weights_local + (1-tau)*actor_weights_target )

        critic_weights_local = np.array( self.critic_local.get_weights(), dtype=object )
        critic_weights_target = np.array( self.critic_target.get_weights(), dtype=object )
        self.critic_target.set_weights( tau*critic_weights_local + (1-tau)*critic_weights_target )

    def hard_update_nets(self):
        self.actor_target.set_weights( self.actor_local.get_weights() )
        self.critic_target.set_weights( self.critic_local.get_weights() )


    # Take action according to epsilon-greedy-policy:
    def action(self, state, epsilon=0.01):

        if random.random < epsilon:
            action_size = 2
            action = 2 * np.random.random_sample(action_size) - 1.0
        else:
            action = self.local_net.predict(state)
            
        return action

    def random_action(self):
        action_size = 2
        action = 2 * np.random.random_sample(action_size) - 1.0
        return action


    def load_weights(self, path):
        filepath = os.path.join(path, "actor_weights_latest.ckpt")
        print("Loading actor network weights from", filepath)
        self.actor_local_net.load_weights(filepath)
        filepath = os.path.join(path, "critic_weights_latest.ckpt")
        print("Loading critic network weights from", filepath)
        self.critic_local_net.load_weights(filepath)
        
        self.hard_update_nets()

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

