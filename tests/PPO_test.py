import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import sys
sys.path.insert(0, '../gymnasium_envs/.')
from PublicCivilityGame import *

t = 105000
DELTA = 0.001
W = 0.7 + DELTA

env = PublicCivilityGame()
env.setWeights([1-W,W])

model = PPO("MlpPolicy", env , verbose=1)
model.learn(total_timesteps=t)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

env = PublicCivilityGame()
env.setWeights([1-W,W])
st , _ = env.reset()

while True:

  at, _ = model.predict(st, deterministic=True)

  st_n, _ , done ,  _ , _ = env.step(at)
  st = st_n
  env.render()

  if done :
    env.render()
    st , _ = env.reset()
