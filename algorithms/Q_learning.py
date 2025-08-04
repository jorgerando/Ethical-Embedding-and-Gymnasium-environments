import random

EPSILON = 0.5
ALFA = 0.7

def best_action(env,Q,st):

    best_value , best_action = float('-inf') , None

    for at in range(env.action_space.n):

       action_value = Q.get( (st, at ) , 0.0 )

       if action_value > best_value :
          best_value = action_value
          best_action = at

    return best_action , best_value  # at , Q(at,st)

def Q_learning(env,WS,gamma=0.7,iterations=100000):

    Q = {}

    env.setWeights(WS)
    st , _ = env.reset()

    for _ in range(iterations):

        if random.random() < EPSILON :
            at , _ = best_action(env,Q,st) # greddy
        else :
            at = env.action_space.sample() # random

        at = int(at)

        st_n, r, done ,  _ , _ = env.step(at)

        _ , action_value = best_action(env,Q,st_n)

        Q[(st,at)] = (1-ALFA)* Q.get( (st, at) , 0.0) + ALFA * (r + gamma * action_value )

        st = st_n

        if done :
          st , _ = env.reset()
          #print("Fin del episodio")

    return Q

def get_pi(env,Q):

    pi = {}

    for st , _ in Q.keys():
       a , _ = best_action(env,Q,st)
       pi[st] = a

    return pi
