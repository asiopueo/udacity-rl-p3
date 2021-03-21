from unityagents import UnityEnvironment
import numpy as np
import time

#################################
#   Initialization:
#################################
env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

#from agent import Agent
from collections import namedtuple, deque

# Reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

state_size = env_info.vector_observations.shape[1]
print('Size of each state:', state_size)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# Initialize the agent:
from agent_torch import MultiAgent
multi_agent = MultiAgent(state_size=state_size, action_size=action_size, buffer_size=10000, batch_size=256, gamma=0.99, learn_rate=0.0001)
multi_agent.load_weights("./checkpoints_torch")



####################################
#  Play loop:
####################################

# Initial values:

def play(n_episodes=300):       
    score_list = []
    score_queue = deque(maxlen=100)
    score_trailing_avg_list = []

    for episode in range(0, n_episodes):
        ticks = 0
        scores = np.zeros( shape=(num_agents,) )

        env_info = env.reset(train_mode=False)[brain_name]   # Reset the environment
        states = env_info.vector_observations               # Get the current state

        multi_agent.reset() # Reset the noise process

        start = time.time()
        while True:
            # Select action according to policy:
            actions = multi_agent.action(states, episode, add_noise=False)
            #print (actions)
            env_info = env.step(actions)[brain_name]
            
            rewards = np.array( env_info.rewards )
            next_states = env_info.vector_observations
            dones = env_info.local_done

            scores += rewards
            states = next_states
            
            if np.any(dones):
                break

            ticks += 1

        end = time.time()

        # Only the agent with the max score counts per episode:
        score_max = np.max(scores)
        score_list.append(score_max)
        score_queue.append(score_max)

        score_trailing_avg = np.mean(score_queue)
        score_trailing_avg_list.append(score_trailing_avg)

        print("***********************************************")
        print("Maximum score of episode {}: {:.2f}".format(episode, score_max))
        print("Trailing avg. score: {:.2f}".format(score_trailing_avg))
        print("Time consumed: {:.2f} s".format(end-start))
        print("***********************************************")

        episode += 1

    return score_list, score_trailing_avg_list


score_list, score_trailing_avg_list = play(300)



env.close()