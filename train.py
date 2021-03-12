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

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# Define named tuple 'Experience'; you can use a dictionary alternatively
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# Initialize the agent:
from agent_torch import Agent
multi_agent = MultiAgent(state_size=, action_size=, buffer_size=10000, batch_size=64, gamma=0.98)




####################################
#  Main learning loop:
####################################


def training(n_episodes=300):       
    tick = 0

    score_list = []
    score_queue = deque(maxlen=100)
    score_trailing_avg_list = []

    #agent.load_weights("./checkpoints")

    for episode in range(0, n_episodes):
        ticks = 0
        scores = np.zeros(num_agents)

        env_info = env.reset(train_mode=True)[brain_name]   # Reset the environment
        state = env_info.vector_observations                # Get the current state

        multi_agent.reset() # Reset the noise process

        start = time.time()
        while True:
            # Select action according to policy:
            actions = multi_agent.action(state, )
            env_info = env.step(action)[brain_name]
            
            rewards = env_info.rewards
            next_states = env_info.vector_observations
            dones = env_info.local_done

            # Add experience to the agent's replay buffer:
            exp = Experience(states, actions, rewards, next_states, dones)
            multi_agent.replay_buffer.insert_into_buffer( exp )
            
            agent.learn()

            scores += rewards
            states = next_states
            
            if np.any(dones) is True:
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
        print("Maximum score of episode {}: {}".format(episode, score_max))
        print("Trailing avg. score: {:.2f}".format(score_trailing_avg))
        print("Time consumed: {:.2f} s".format(end-start))
        print("***********************************************")


        print("Total score:", score)
        agent.save_weights("./checkpoints")

        episode += 1

    return score_list, score_trailing_avg_list


training(30)





env.close()


