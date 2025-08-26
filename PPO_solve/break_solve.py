import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

import sys
sys.path.insert(0, '../gymnasium_envs/.')
from BreakableBottles import *

t = 80000
GAMMA = 1.0
DELTA = 0
W = 0.0 + DELTA

env = GymBreakableBottles()
env.setWeights([1-W,W])

model = PPO("MlpPolicy", env , gamma=GAMMA , verbose=1)

model.learn(total_timesteps=t  )
model.save("./etical_policys/ppo_break_pi.zip")

del model

model = PPO.load("./etical_policys/ppo_break_pi.zip")

env = GymBreakableBottles()
env.setWeights([1-W,W])
st , _ = env.reset()

while True:

  env.render()
  at, _ = model.predict(st, deterministic=True)
  at = int(at)

  st_n, _ , done ,  _ , _ = env.step(at)
  st = st_n

  if done :
    env.render()
    st , _ = env.reset()
