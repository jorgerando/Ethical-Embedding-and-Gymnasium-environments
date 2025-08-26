import sys

sys.path.insert(0, '../gymnasium_envs/.')
from PublicCivilityGame import *

sys.path.insert(0, '../algorithms/.')
from Q_learning import *
from OLS_Embding import *

ITERATIONS = 200000
GAMMA = 0.7
EPSILON = 0.5
ALFA = 0.7
EPISODES = 1

if __name__ == '__main__':

    env = PublicCivilityGame()

    #Calculo del convex hull
    S = OLS3(env ,  gamma=GAMMA , epsilon=EPSILON , alfa=ALFA , iterations=ITERATIONS ,episodes=EPISODES )

    # extraccion de las politicas eticas obtimas
    S_ = ethical_optimal_extraction(S)
    print("HULL (V1,V2) : " + str(S_) )

    # calculo del peso etico
    w = ethical_embedding_state(S_)
    print("Peso etico :" , w)

    #### EJECUCION DEL ENTORNO ####

    Q , _ = Q_learning(env,[1-w,w],gamma=GAMMA)
    pi = get_pi(env,Q)

    st , _ = env.reset()

    while True:
        env.render()
        at = pi[tuple(st)]
        st_n, r, done ,  _ , _ = env.step(at)

        st = st_n
        if done :
            env.render()
            st , _ = env.reset()
