import random
import queue as Q_
from Q_learning import *

def get_V0s(env,pi,WS,gamma=0.7,episodes=25):

    s , _ = env.reset()
    env.setWeights(WS)

    RI_T = 0
    RE_T = 0

    for _ in range(episodes):

     RI = 0
     RE = 0
     C = 1
     done = False

     while not done :

        a = pi[s]
        s , _ , done , _ , info = env.step(a)

        r1 = info['Individual']
        r2 = info['Etical']

        RI += gamma**(C-1) * r1
        RE += gamma**(C-1) * r2

        C+=1
        #print("u")

        if done :
            #print("Fin del episodio")
            RI_T += RI
            RE_T += RE
            s , _ = env.reset()
            env.setWeights(WS)
            #print("fin")
            break

    return RI_T/episodes , RE_T/episodes

def new_weight(v,S):

    # max ( (1-W2_i)*V1(s0) + (W2_i)*V2(s0) )

    x = -1
    for v_prima in S:
        if v[0]==v_prima[0] and v[1]==v_prima[1]:
            continue
        else:
            new_x = (v[0] - v_prima[0]) / (v[0] - v[1] - v_prima[0] + v_prima[1])
        if new_x >= x:
           x = new_x

    weight_vector = [1-x, x]

    return  weight_vector

def OLS3(env,gamma=0.7, iterations=1000000 , episodes=25 ):
    S = []
    W = []
    print(gamma)

    q = Q_.PriorityQueue()
    q.put((-9999, [1, 0]))
    q.put((-9999, [0.01, 0.99 ]))  # <---- Por que tendria que ser [0,1]

    print(">> Iniciando OLS3...")

    while not q.empty():
        weight_vector = q.get()[1]
        print(f"\nEvaluando vector de pesos: {weight_vector}")

        Q = Q_learning(env,weight_vector,gamma,iterations)
        pi = get_pi(env,Q)
        v1, v2 = get_V0s(env, pi, weight_vector,gamma,episodes)

        vs = [v1, v2]
        W.append(weight_vector)
        S.append((v1, v2))

        print(f"-> V0s: {vs}")

        w = new_weight(vs, S)

        if w[1] != weight_vector[1] and w[1] > 0:
            print(f"-> Nuevo vector generado: {w}")
            q.put((-999, w))

    print("\n>> OLS3 finalizado.")
    return S

def ethical_optimal_extraction(S):
    S = list(dict.fromkeys(S))
    return sorted(S, key=lambda x: x[1])

def ethical_embedding_state(hull):
    #
    w = 0.0

    if len(hull) < 2:
        print( "ENBOLVENTE : ",hull)
        return w
    else:
        ethically_sorted_hull = hull

        best_ethical = ethically_sorted_hull[-1]
        second_best_ethical = ethically_sorted_hull[-2]

        individual_delta = second_best_ethical[0] - best_ethical[0]
        ethical_delta = best_ethical[1] - second_best_ethical[1]

        if ethical_delta != 0:
            w = individual_delta/ethical_delta

        return w
