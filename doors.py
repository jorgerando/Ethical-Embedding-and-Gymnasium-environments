import sys
sys.path.insert(0, 'algorithms/.')
from Q_learning import *
from OLS_Embding import *

from gymnasium_envs.Doors import *

if __name__ == '__main__':

    ITERATIONS = 80000
    GAMMA = 1.0
    EPISODES = 10

    env = GymDoors()
    #Calculo del convex hull
    S = OLS3(env , gamma=GAMMA, iterations=ITERATIONS ,episodes=EPISODES)

    # extraccion de las politicas eticas obtimas
    S_ = ethical_optimal_extraction(S)
    print("HULL (V1,V2) : " + str(S_) )

    # calculo del peso etico
    w = ethical_embedding_state(S_)
    w  += 0.001
    print("Peso etico :" , w)

    #w = 2.1 PESOS DE MANEL

    Q = Q_learning(env,[1-w,w],gamma=GAMMA)
    pi = get_pi(env,Q)

    st , _ = env.reset()
    actions = []

    print("Peso etico :" , w)
    env.render()

    while True:

      at = pi[st]
      str_a = env.action2string[int(at)]
      actions.append(str_a)
      st_n, r, done ,  _ , _ = env.step(at)

      st = st_n
      env.render()

      if done :
        env.render()
        st , _ = env.reset()
        print("Peso etico :" , w)
        print("Secuencia de acciones :",actions)
        break
