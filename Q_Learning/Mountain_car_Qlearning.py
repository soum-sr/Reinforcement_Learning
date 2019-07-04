import gym
import numpy as np 
import matplotlib.pyplot as plt 

env = gym.make("MountainCar-v0")

"""
Here the the state is : position , velocity
and it we can control it with 3 actions: push left, do nothing, push right 

"""
### HYPERPARAMETERS
LEARNING_RATE = 0.01
DISCOUNT = 0.95
EPISODES = 5000
SHOW_EVERY = 1000

### SETTING UP DIMENSIONS
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Epsilong will make sure the model explores first then exploits
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high =0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

ep_rewards = []
aggregate_ep_rewards = {'ep':[], 'avg': [], 'min':[], 'max':[]}



### WE NEED TO CONVERT THE CONTINUOUS STATES TO DISCRETE STATES
def convert_cont_disc(state):
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):

	total_reward = []
	episode_reward = 0

	if episode % SHOW_EVERY == 0: 
		print(f"EPISODE: {episode}")
		render = True
	else:
		render = False

	discrete_state = convert_cont_disc(env.reset())

	done = False

	while not done:

		if np.random.random() > epsilon:
			action = np.argmax(q_table[discrete_state])
		else:
			action = np.random.randint(0, env.action_space.n)

		new_state, reward, done, _ = env.step(action)
		total_reward.append(reward)
		episode_reward += reward

		new_discrete_state = convert_cont_disc(new_state)
		if render:
			env.render()
		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action, )]
			### BELLMAN'S EQUATION
			new_q = (1-LEARNING_RATE)* current_q + LEARNING_RATE * (reward + DISCOUNT*max_future_q)
			q_table[discrete_state + (action, )] = new_q

		elif new_state[0] >= env.goal_position:
			print(f"We made it on episode {episode}")
			q_table[discrete_state + (action, )] = 0

		discrete_state = new_discrete_state
	if episode % SHOW_EVERY == 0:
		print(f"Reward in episode {episode} is {sum(total_reward)}")

	if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
		epsilon -= epsilon_decay_value
	ep_rewards.append(episode_reward)

	if not episode % SHOW_EVERY:
		average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
		aggregate_ep_rewards['ep'].append(episode)
		aggregate_ep_rewards['avg'].append(average_reward)
		aggregate_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
		aggregate_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

		print(f"Episode: {episode} Avg: {average_reward} Min: {min(ep_rewards[-SHOW_EVERY:])} Max: {max(ep_rewards[-SHOW_EVERY:])}")

env.close()

plt.plot(aggregate_ep_rewards['ep'], aggregate_ep_rewards['avg'], label = "avg")
plt.plot(aggregate_ep_rewards['ep'], aggregate_ep_rewards['min'], label = "min")
plt.plot(aggregate_ep_rewards['ep'], aggregate_ep_rewards['max'], label = "max")
plt.legend(loc=4)
plt.show()