from unityagents import UnityEnvironment
import numpy as np

# Initialization of Unity environment
env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


from Agent import MultiAgent
from collections import namedtuple

# Initialize the agent:
multi_agent = MultiAgent(buffer_size=1000, batch_size=20, gamma=0.98)

# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# Size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# Examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# Define named tuple 'Experience'; you can use a dictionary alternatively
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


# Initial values:
state = env_info.vector_observations[0] # get the current state
score = 0                               # Score is NOT the discounted reward but the final 'Banana Score' of the game
time = 0


#################################
#   Play one episode:
#################################
def play_one_turn():
    global score, time, state, env_info

    # Select action according to policy:
    # action = ...
    # Select random action:
    action = multi_agent.action(state)

    print('Action taken: ', action, 'Time: ', time)

    # Take action and record the reward and the successive state
    env_info = env.step(action)[brain_name]

    reward = env_info.rewards[0]
    next_state = env_info.vector_observations[0]
    done = env_info.local_done[0] # Not really relevant in this experiment as it runs 300 turns anyway

    # Add experience to the agent's replay buffer:
    exp = Experience(state, action, reward, next_state, done)
    multi_agent.replay_buffer.insert_into_buffer( exp )

    # If buffer is sufficiently full, let the agent learn from his experience:
    #if agent.replay_buffer.buffer_usage():
    #    agent.learn()

    score += reward
    state = next_state



while True:
    play_one_turn()
    
    if time%50 == 0:
        print("[Time: {}] Score".format(time))
    elif time%10 == 0:
        print("[Time: {}] Time to update the target net.".format(time))
        print("Buffer usage: {}".format(ma.replay_buffer.buffer_usage()))
        #agent.update_target_net()

    time += 1



