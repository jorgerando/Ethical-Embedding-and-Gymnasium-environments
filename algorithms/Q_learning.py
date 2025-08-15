import random
import collections

def best_action(env,Q,st):

    best_value , best_action = float('-inf') , None

    for at in range(env.action_space.n):

       action_value = Q.get( (st, at ) , 0.0 )

       if action_value > best_value :
          best_value = action_value
          best_action = at

    return best_action , best_value  # at , Q(at,st)

def Q_learning(env,WS,gamma=0.7,epsilon=0.5,alfa=0.7,iterations=100000):

    Q = {} #<-----

    try:
       env.setWeights(WS)
    except Exception as e:
      pass # el entorno no tiene setWeights

    st , _ = env.reset()

    for i in range(iterations):

        if random.random() < epsilon :
          at = env.action_space.sample()  # random
        else:
          at, _ = best_action(env, Q, st) # explotar

        at = int(at)

        st_n, r, d , tr , _ = env.step(at)

        done = d or tr

        if done:
            target = r
        else:
            _, action_value = best_action(env, Q, st_n)
            target = r + gamma * action_value

        Q[(st,at)] = (1-alfa)* Q.get( (st, at) , 0.0) + alfa * (target)

        st = st_n

        if i % 1000 == 0 :
            print("Iteracion : " + str(i) )

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

def get_V0(env,pi,WS,gamma=0.7,episodes=25):

    s , _ = env.reset()
    R_t = 0

    for i in range(episodes):

     R = 0
     C = 1
     done = False
     print(i)

     while not done :

        a = pi[s]
        s , r , t , tr , _ = env.step(a)

        #R += gamma**(C-1) * r
        R+=r
        C+=1

        done = t or tr

        if done :
            print(R)
            R_t += R

            s , _ = env.reset()
            break

    return R_t/episodes

##### Modular implementation ####

def best_action2(env,Q,s):

        best_value, best_action = None, None

        for a in range(env.action_space.n) :

            value = Q[(s, a)]

            if best_value is None or value > best_value :
                best_value = value
                best_action = a

        return best_action , best_value

def play_episode(env,Q,WS=[1,1]):

    try:
       env.setWeights(WS)
    except Exception as e:
      pass # el entorno no tiene setWeights

    total_reward = 0.0
    state , _ = env.reset()

    while True:
        action , _ = best_action2(env,Q,state)
        new_state, reward, is_done, tr , _ = env.step(action)
        total_reward += reward
        if is_done or tr :
            break
        state = new_state

    return total_reward

def V0(env,Q,epi=25,WS=[1,1]):
    v0 = 0
    for _ in range(epi):
        v0 += play_episode(env,Q,WS)
    return v0 / epi

def Q_learning2(env,WS,gamma=0.7,epsilon=0.5,alfa=0.7,iterations=100000):

    Q = collections.defaultdict(float) # diccionario normal solo que si no esta la clave devuelve 0 siempre
    info = []
    best_v0 = -10000000000
    best_Q = None

    try:
       env.setWeights(WS)
    except Exception as e:
      pass # el entorno no tiene setWeights

    s0 , _ = env.reset()
    s = s0

    for i in range(iterations):

        if random.random() > epsilon :
            a , _ = best_action2(env,Q,s)
        else :
            a = env.action_space.sample()

        sn , r  , done , tr , _ = env.step(a)

        _ , value = best_action2(env,Q,sn)

        Q[(s,a)] = Q[(s,a)] + alfa * ( r + gamma * value - Q[(s,a)] )

        s = sn

        if done or tr :

            v0 = V0(env,Q,WS=WS)
            info.append(v0)

            if v0 > best_v0 or best_Q == None :
                best_v0 = v0
                best_Q = Q.copy()
                print("Best renward:",best_v0,"iter:",i)

            s , _ = env.reset()


    return best_Q , info

def get_pi2(env,Q):

    pi = {}

    for st , _ in Q.keys():
       a , _ = best_action2(env,Q,st)
       pi[st] = a

    return pi
