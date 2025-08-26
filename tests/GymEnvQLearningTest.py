import sys
sys.path.insert(0, '..')
from gymnasium_envs.UnbreakableBottles  import *
from gymnasium_envs.BreakableBottles  import *
sys.path.insert(0, '../algorithms/.')
from Q_learning import *
from OLS_Embding import *
import gymnasium as gym
import time
import random

import collections

ITERATIONS = 10000
EPISODES = 100
GAMMA = 0.9
EPSILON = 0.5
ALFA = 0.2
WS = [None,None]

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True )

Q , _ = Q_learning(env,WS,gamma=GAMMA,epsilon=EPSILON,alfa=ALFA,iterations=ITERATIONS)
pi = get_pi(env,Q)

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True ,render_mode="human")
state , _ = env.reset()

while True:
    action = pi[state]
    #action = int(action)
    env.render()
    time.sleep(0.1)
    new_state, reward, is_done, tr , _ = env.step(action)

    if is_done or tr :
        env.render()
        state , _ = env.reset()

    state = new_state
