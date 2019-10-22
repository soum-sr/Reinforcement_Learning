# Reinforcement_Learning
------------------------
## Q-learning Algorithm

The 'Q' in the Q-Learning stands for Quality. It represents how effective a given action is in gaining some future reward. Here we create a Q-Table that returns the best actions to take at certain scenario or state. It has the shape of [state, action]. Initially it is started with zeros then after each training iteration the value is updated. And after the model is completely trained we take reference from the Q-Table and perform our action for a given state.
