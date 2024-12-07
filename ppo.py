from functional import *

ppo=agent('PPO','MlpPolicy')
ppo.learn()

ppo.evaluate()
ppo.visualize()