import sys

sys.path.insert(0, '../gymnasium_envs/.')
from PublicCivilityGame import *

sys.path.insert(0, '../algorithms/.')
from Q_learning import *
from OLS_Embding import *

import pickle

ITERATIONS = 200000
GAMMA = 0.7
EPSILON = 0.5
ALFA = 0.7
EPISODES = 1

DELTA = 0.001
W = 0.7 + DELTA #### Peso calculado previamente
pi = None

# V0 (sin gamma importante ) : 43.2

if __name__ == '__main__':

    '''
    env = PublicCivilityGame()
    Q , _ = Q_learning(env,[1-W,W],gamma=GAMMA,epsilon=EPSILON,alfa=ALFA,iterations=ITERATIONS,test_episodes=EPISODES)
    pi = get_pi(env,Q)

    with open('etical_policys/civity_game_pi.pkl', 'wb') as fichero:
     pickle.dump(pi, fichero)
    '''
    
    with open('etical_policys/civity_game_pi.pkl', 'rb') as archivo:
      pi = pickle.load(archivo)

    env = PublicCivilityGame()
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
