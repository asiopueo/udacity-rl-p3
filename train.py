from unityagents import UnityEnvironment
import numpy as np
import time



#################################
#  Initialization:
#################################
env = UnityEnvironment(file_name="./Reacher_Linux_1/Reacher.x86_64")
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
agent = Agent(buffer_size=10000, batch_size=64, gamma=0.98, epsilon=0.1, action_size=4)




####################################
#  Main learning loop:
####################################

# Initial values:

episode = 0
score = 0           
tick = 0

score_list = []
score_trailing_list = deque(maxlen=10)

eps = 1.0
eps_rate = 0.995
eps_end = 0.02


#agent.load_weights("./checkpoints")

for episode in range(0, 300):
    ticks = 0
    score = 0

    env_info = env.reset(train_mode=True)[brain_name]  # Reset the environment
    state = env_info.vector_observations[0]             # Get the current state

    start = time.time()
    while True:
        # Select action according to policy:
        action = agent.action(state, eps, add_noise=True)

        print('Action taken: ', action, 'Time: ', tick)

        # Take action and record the reward and the successive state
        env_info = env.step(action)[brain_name]
        
        reward = env_info.rewards[0]
        next_state = env_info.vector_observations[0]
        done = env_info.local_done[0]

        # Add experience to the agent's replay buffer:
        exp = Experience(state, action, reward, next_state, done)
        agent.replay_buffer.insert_into_buffer( exp )
        
        agent.learn()

        score += reward
        state = next_state
        
        eps = max( eps_rate*eps, eps_end )

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
    print("Greedy epsilon used: {}".format(eps))
    print("Time consumed: {:.2f} s".format(end-start))
    print("***********************************************")


    print("Total score:", score)
    agent.save_weights("./checkpoints")

    episode += 1

env.close()


