import sys

sys.path.insert(0, '../algorithms/.')
from Q_learning import *

sys.path.insert(0, '../gymnasium_envs/.')
from Doors import *

import pickle

ITERATIONS = 100000
GAMMA = 1.0
EPSILON = 0.5
ALFA = 0.7
EPISODES = 1

DELTA = 0.001
W = 0.1 + DELTA #### Peso calculado previamente

pi = None

if __name__ == '__main__':
    '''
    env = GymDoors()
    Q , _ = Q_learning(env,[1-W,W],gamma=GAMMA,epsilon=EPSILON,alfa=ALFA,iterations=ITERATIONS,test_episodes=EPISODES)
    pi = get_pi(env,Q)

    with open('etical_policys/doors_pi.pkl', 'wb') as fichero:
     pickle.dump(pi, fichero)
    '''
    with open('etical_policys/doors_pi.pkl', 'rb') as archivo:
      pi = pickle.load(archivo)

    env = GymDoors()
    env.setWeights([1-W,W])
    st , _ = env.reset()

    while True:

      env.render()
      at = pi[tuple(st)]
      st_n, _ , done ,  _ , _ = env.step(at)

      st = st_n

      if done :
        env.render()
        st , _ = env.reset()
