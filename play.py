from unityagents import UnityEnvironment
import numpy as np
import time

#################################
#  Initialization:
#################################
env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

#from agent import Agent
from collections import namedtuple, deque
import time

# Reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# Define named tuple 'Experience'; you can use a dictionary alternatively
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# Initialize the agent:
from agent_torch import MultiAgent
state_size = 8
multi_agent = MultiAgent(state_size, action_size, buffer_size=10000, batch_size=64, gamma=0.98)



####################################
#  Main learning loop:
####################################

# Initial values:

episode = 0
score = 0           
tick = 0

score_list = []
score_trailing_list = deque(maxlen=100)


#agent.load_weights("./checkpoints")

for episode in range(0, 300):
    ticks = 0
    score = 0

    env_info = env.reset(train_mode=False)[brain_name]  # Reset the environment
    state = env_info.vector_observations[0]             # Get the current state

    start = time.time()
    while True:
        # Select action according to policy:
        #action = multi_agent.action(state, eps)
        action = multi_agent.random_action()

        # Take action and record the reward and the successive state
        env_info = env.step(action)[brain_name]
        
        reward = env_info.rewards[0]
        next_state = env_info.vector_observations[0]
        done = env_info.local_done[0]

        score += reward
        state = next_state
        
        if done is True:
            break

        ticks += 1


    end = time.time()

    score_list.append(score)
    score_trailing_list.append(score)

    score_avg = np.mean(score_list)
    score_trailing_avg = np.mean(score_trailing_list)

    print("***********************************************")
    print("Score of episode {}: {}".format(episode, score))
    print("Avg. score: {:.2f}".format(score_avg))
    print("Trailing avg. score: {:.2f}".format(score_trailing_avg))
    print("Time consumed: {:.2f} s".format(end-start))
    print("***********************************************")

    episode += 1

env.close()

env.close()