import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
env.reset()

discrete_obs_space_size = [30] * len(env.observation_space.high)
print(discrete_obs_space_size)

discrete_obs_window_size = (env.observation_space.high - env.observation_space.low)/discrete_obs_space_size
print(discrete_obs_window_size)

q_table = np.random.uniform(low=-2, high=0, size=(discrete_obs_space_size + [env.action_space.n]))
print("Shape of q_table: ", q_table.shape)

LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.95 
EPISODES = 5000

ep_rewards = []
aggr_ep_rewards = {'episode':[], 'avg':[], 'min':[], 'max':[]}
# Render the env for every 2000th episode
SHOW_ENV = 500

## Handling exploration and exploitation
epsilon = 0.5
start_epsilon_decay = 1
end_epsilon_decay = EPISODES // 2
epsilon_decay_val = epsilon/(end_epsilon_decay - start_epsilon_decay)

def convert_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_obs_window_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    
    episode_reward = 0
    

    if episode % SHOW_ENV == 0:
        print("Episode: ", episode)
        # save the qtable
        np.save("q_tables/{}-qtable.npy".format(episode), q_table)
        render_env = True
    else:
        render_env = False
        
    discrete_state = convert_discrete_state(env.reset())
    
    complete = False

    while not complete:
        # Random float between 0 and 1
        # If greater we'll do exploitation
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        # else we'll choose a random action(exploration)
        else:
            action = np.random.randint(0, env.action_space.n)
        
        
        new_state, reward, complete, _ = env.step(action)
        
        # updating the episode reward
        episode_reward += reward

        new_discrete_state = convert_discrete_state(new_state)
        
        #Render the environment 
        if render_env:
            env.render()

        if not complete:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )] 

            # Bellman's equation
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_RATE * max_future_q)

            # update q_table
            q_table[discrete_state + (action, )] = new_q
        # If the new position of the car is more than or equal to the goal position then we 
        # do not give any punishment to the agent
        elif new_state[0] >= env.goal_position:
            print("Car reached the top at episode:", episode)
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state
    
    if end_epsilon_decay >= episode >= start_epsilon_decay:
        epsilon -= epsilon_decay_val
    
    # appending the total reward of the episode
    ep_rewards.append(episode_reward)
    
    if not episode % SHOW_ENV:
        average_reward = sum(ep_rewards[-SHOW_ENV:])/len(ep_rewards[-SHOW_ENV:])
        aggr_ep_rewards['episode'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_ENV:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_ENV:]))
        
        print("Episode: {} avg: {} min: {} max: {}".format(episode,average_reward,min(ep_rewards[-SHOW_ENV:]),min(ep_rewards[-SHOW_ENV:])))
        
env.close()  

plt.plot(aggr_ep_rewards['episode'], aggr_ep_rewards['avg'], label = "avg")
plt.plot(aggr_ep_rewards['episode'], aggr_ep_rewards['min'], label = "min")
plt.plot(aggr_ep_rewards['episode'], aggr_ep_rewards['max'], label = "max")
plt.legend(loc=4)
plt.show()
