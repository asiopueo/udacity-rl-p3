from collections import deque
import random
import numpy as np
from collections import namedtuple

import Network




class Agent():
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
        self.actor_net = Network.network_actor()
        self.critic_net = Network.network_critic()

    # Let the agent learn from experience
    def learn(self):
        # Retrieve batch of experiences from the replay buffer:
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample_from_buffer()
        # Prepare the target. Note that Q_target[:,action] will need to be assigned the 'true' learning target.
        Q_target = self.local_net.predict( state_batch )
        Q_next_state = np.max( self.target_net.predict(next_state_batch), axis=1 )

        X = []
        y = []

        # Batches need to be prepared before learning
        for index, state in enumerate(state_batch):    
            # Calculate the next q-value according to SARSA-MAX   
            # Q_new w.r.t. action:
            if not done_batch[index]:
                Q_new = reward_batch[index] + self.gamma * Q_next_state[index]
            else:
                Q_new = reward_batch[index]

            Q_target[index, action_batch[index]] = Q_new

            X.append(state)
            y.append(Q_target[index])

        X_np = np.array(X)
        y_np = np.array(y)

        print("X_np.shape: ", X_np.shape)
        print("y_np.shape: ", y_np.shape)

        self.local_net.fit(X_np, y_np, batch_size=self.batch_size, epochs=1, shuffle=False, verbose=1)



    # Take action according to epsilon-greedy-policy:
    def action(self, state, epsilon=0.9):
        if random.random() > epsilon:
            return random.randrange(0,4)
        else:
            return self.local_net(state)
        
        prob_distribution = self.local_net.predict(state.reshape(1,-1))
        action = np.argmax(prob_distribution)
        return action

    # Copy weights from short-term model to long-term model
    def update_target_net(self, tau=0.1):
        # Implement soft update for later:
        # get_weights()[0] -- weights
        # get weights()[1] -- bias (if existent)
        # Soft-update:
        self.actor_net.set_weights( tau*self.actor_net.get_weights() + (1-tau)*self.critic_net.get_weights() )



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
