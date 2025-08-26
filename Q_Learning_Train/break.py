import sys
sys.path.insert(0, '../algorithms/.')
from Q_learning import *
from OLS_Embding import *

sys.path.insert(0, '../gymnasium_envs/.')
from BreakableBottles import *

ITERATIONS = 1000000
GAMMA = 0.9
EPSILON = 0.5
ALFA = 0.25
EPISODES = 10

if __name__ == '__main__':

    env = GymBreakableBottles()
    #Calculo del convex hull
    S = OLS3(env , gamma=GAMMA , epsilon=EPSILON , alfa=ALFA, iterations=ITERATIONS ,episodes=EPISODES)

    # extraccion de las politicas eticas obtimas
    S_ = ethical_optimal_extraction(S)
    print("HULL (V1,V2) : " + str(S_) )

    # calculo del peso etico
    w = ethical_embedding_state(S_)
    w+=0.01

    env = GymBreakableBottles()

    print("Peso etico :" , w)

    Q , _ = Q_learning(env,[1-w,w],gamma=GAMMA , epsilon=EPSILON , alfa=ALFA)
    pi = get_pi(env,Q)

    st , _ = env.reset()
    actions = []

    env.render()
    while True:

      at = pi[tuple(st)]
      str_a = env.action2string[int(at)]
      actions.append(str_a)
      st_n, r, done ,  _ , _ = env.step(at)

      st = st_n

      print("Peso etico :" , w)
      env.render()

      if done :
        env.render()
        st , _ = env.reset()
        print("Peso etico :" , w)
        print("Secuencia de acciones :",actions)
        actions = []
