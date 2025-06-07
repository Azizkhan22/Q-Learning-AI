import numpy as np
import gymnasium as gym
import time
import pygame
# Create the environment
env = gym.make("FrozenLake-v1",is_slippery=False)

# Initialize Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n 
Q = np.zeros((num_states, num_actions))

alpha = 0.9
gamma = 0.8
epsilon = 0.6
episodes = 1000
# Training the AI agent
for episode in range(episodes):
    state, info = env.reset()
    done = False
    while not done:
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])  
        
        next_state, reward, done, truncated, info = env.step(action)

        # Q-learning update
        td = reward + gamma * Q[next_state,np.argmax(Q[next_state])] - Q[state, action]
        # print(state,next_state,action,reward,td,done)
        Q[state, action] += alpha * td
        state = next_state

print("Q-table after training:")
print(Q)

# Function to visualize the Frozen Lake environment
def visualize_environment(env):
    env.render()
    time.sleep(0.0001)

env = gym.make("FrozenLake-v1",render_mode="human",is_slippery=False)

# Number of episodes to run for visualization
num_episodes = 0

for episode in range(3):
    state, info = env.reset()
    done = False
    print(f"Episode: {episode + 1}")
    visualize_environment(env)

    while not done and num_episodes < 100:
        action = np.argmax(Q[state])  # Choose action based on the learned Q-table
        print(f"State: {state}, Action: {action}, Q-values: {Q[state]}")
        next_state, reward, done, truncated, info = env.step(action)
        
        # Visualize the environment
        visualize_environment(env)
        
        state = next_state
        num_episodes = num_episodes + 1

    if reward == 1:
        print("Reached the goal!")
    else:
        print("Fell into a hole or did not reach the goal.")

env.close()
