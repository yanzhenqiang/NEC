# import gym
# env = gym.make('PongNoFrameskip-v4')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         # env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()

import numpy as np

b = np.expand_dims(a, axis=0) 
b = [[[1, 2, 3], [4, 5, 6]]]
b.shape = (1, 2, 3)