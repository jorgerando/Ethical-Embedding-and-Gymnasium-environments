import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

import sys
sys.path.insert(0, '../gymnasium_envs/.')
from Doors import *

t = 500000
GAMMA = 0.99
DELTA = 0.001
W = 0.1 + DELTA

env = GymDoors(WS=[1-W,W])
env.setWeights([1-W,W])

model = PPO("MlpPolicy", env , gamma=GAMMA ,ent_coef=0.25 , verbose=1)


eval_callback = EvalCallback(
    env,
    best_model_save_path="./etical_policys/",
    eval_freq=1000,
    deterministic=True,
    render=False
)


model.learn(total_timesteps=t  ,  callback=eval_callback )
model.save("./etical_policys/ppo_door_pi.zip")

del model

model = PPO.load("./etical_policys/best_model.zip")

env = GymDoors()
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
