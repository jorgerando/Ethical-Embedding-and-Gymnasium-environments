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

ITERATIONS = 3000000
EPISODES = 100
GAMMA = 0.9
EPSILON = 0.5
ALFA = 0.2

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True )
env = GymBreakableBottles()
env.setWeights([1,1])

Q  = collections.defaultdict(float)

def best_action(s):

    best_value, best_action = None, None

    for a in range(env.action_space.n) :

        value = Q[(s, a)]

        if best_value is None or value > best_value :
            best_value = value
            best_action = a

    return best_action , best_value

def play_episode(env):
    total_reward = 0.0
    state , _ = env.reset()
    while True:
        action , _ = best_action(state)
        #action = int(action)
        #env.render()
        new_state, reward, is_done, tr , _ = env.step(action)
        total_reward += reward
        if is_done or tr :
            break
        state = new_state
    return total_reward

def V0(epi=10):
    v0 = 0
    for _ in range(epi):
        v0 += play_episode(env)
    return v0 / epi

s , _ = env.reset()

i = 0
renward_best = -1000000000
ant_renward_best = -10000000
while True :

    if random.random() > EPSILON :
        a , _ = best_action(s)
    else :
        a = env.action_space.sample()

    sn , r  , done , tr , _ = env.step(a)

    _ , value = best_action(sn)

    Q[(s,a)] = Q[(s,a)] + ALFA * ( r + GAMMA * value - Q[(s,a)] )

    s = sn

    if done or tr :
        rn = V0(100)
        if rn > renward_best:
            renward_best = rn
            print("Best Renward: " + str (renward_best) )

        if renward_best > 35 :
            break
        s , _ = env.reset()

state , _ = env.reset()
while True:
    action , _ = best_action(state)
    #action = int(action)
    env.render()
    new_state, reward, is_done, tr , _ = env.step(action)

    if is_done or tr :
        env.render()
        state , _ = env.reset()

    state = new_state
