from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from unityagents import UnityEnvironment
from maddpg_agent import Agent
#%matplotlib inline

# Load Unity Tennis environment
env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64", no_graphics=True)

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

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

# Initialize agents
random_seed = 0
train_mode = True
P1_agent = Agent(state_size, action_size, num_agents=1, random_seed=0)
P2_agent = Agent(state_size, action_size, num_agents=1, random_seed=0)

def maddpg(n_episodes=2000, max_t=1000, print_every=100, goal_score=0.5, episodes=100):
    scores_deque = deque(maxlen=100)
    total_scores = []
    average_score = []
    max_score = -np.Inf
    max_score_episode = 0
    start_time = time.time()

    for i_episode in range(1, n_episodes + 1):
        # Reset environment
        env_info = env.reset(train_mode=train_mode)[brain_name]
        # Get current state for each agent and combine them
        states = np.reshape(env_info.vector_observations, (1,48))
        P1_agent.reset()
        P2_agent.reset()
        # Initialize the score (for each agent)
        scores = np.zeros(num_agents)

        for t in range(max_t):
            # Select an action
            action_P1 = P1_agent.act(states, add_noise=True)  # agent 0 chooses an action
            action_P2 = P2_agent.act(states, add_noise=True)  # agent 1 chooses an action
            actions = np.concatenate((action_P1, action_P2), axis=0).flatten()

            # Send actions to environment
            env_info = env.step(actions)[brain_name]

            # Get next state and combine them
            next_states = np.reshape(env_info.vector_observations, (1, 48))

            # Get reward
            rewards = env_info.rewards

            # Check if episode has finished
            done = env_info.local_done

            # Send actions to the agents
            P1_agent.step(states, actions, rewards[0], next_states, done, 0)  # agent 1 learns
            P2_agent.step(states, actions, rewards[1], next_states, done, 1)  # agent 2 learns

            # Pass states to next time step
            states = next_states
            
            # Update the scores
            scores += np.max(rewards)

            # Exit loop if episode finished
            if np.any(done):
                break

        episode_max_score = np.max(scores)
        scores_deque.append(episode_max_score)
        total_scores.append(episode_max_score)
        average_score.append(np.mean(scores_deque))

        # Save best score
        if episode_max_score > max_score:
            max_score = max_score_episode
            max_score_episode = i_episode

        # Print results
        if i_episode % print_every == 0:
            print('Episodes {:0>4d}-{:0>4d}\t Highest Reward: {:.3f}\t Lowest Reward: {:.3f}\t Average Score: {:.3f}'.format(
                i_episode-print_every, i_episode, np.max(total_scores[-print_every:]), np.min(total_scores[-print_every:]), average_score[-1]))


        # Determine if environment is solved and keep best performing models
        if average_score[-1] >= goal_score:
            print('\nEnvironment solved in {:d} episodes! \
                \nAverage Score: {:.3f} over past {:d} episodes'.format(
                    i_episode-episodes, average_score[-1], episodes))
            print('Training time = {:.5f} sec'.format(time.time() - start_time))

            # Save weights
            torch.save(P1_agent.actor_local.state_dict(), 'checkpoint_actor_P1_agent.pth')
            torch.save(P1_agent.critic_local.state_dict(), 'checkpoint_critic_P1_agent.pth')
            torch.save(P2_agent.actor_local.state_dict(), 'checkpoint_actor_P2_agent.pth')
            torch.save(P2_agent.critic_local.state_dict(), 'checkpoint_critic_P2_agent.pth')
            break

    return total_scores, average_score

# Start training
scores, avgs = maddpg()

# Plot results
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.plot(np.arange(len(scores)), avgs, c='r', label='Average Score avg')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# Close environment
env.close()