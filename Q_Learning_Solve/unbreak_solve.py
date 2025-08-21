import sys
sys.path.insert(0, '../algorithms/.')
from Q_learning import *

sys.path.insert(0, '../gymnasium_envs/.')
from UnbreakableBottles import *

import pickle

ITERATIONS = 10000
GAMMA = 0.9
EPSILON = 0.5
ALFA = 0.7
EPISODES = 100

DELTA = 0.001
W = 0.0 + DELTA #### Peso calculado previamente

pi = None

if __name__ == '__main__':


    env = GymUnbreakableBottles()
    Q , _ = Q_learning(env,[1-W,W],gamma=GAMMA,epsilon=EPSILON,alfa=ALFA,iterations=ITERATIONS,test_episodes=EPISODES)
    pi = get_pi(env,Q)

    with open('etical_policys/unbreak_etical_pi.pkl', 'wb') as fichero:
     pickle.dump(pi, fichero)


    with open('etical_policys/unbreak_etical_pi.pkl', 'rb') as archivo:
      pi = pickle.load(archivo)

    env = GymUnbreakableBottles()
    env.setWeights([1-W,W])

    st , _ = env.reset()

    env.render()
    while True:

      at = pi[tuple(st)]
      st_n, _ , done ,  _ , _ = env.step(at)
      st = st_n
      env.render()

      if done :
        env.render()
        st , _ = env.reset()
