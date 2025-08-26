import random
import collections

def best_action(env,Q,s):

    best_value, best_action = None, None

    for a in range(env.action_space.n) :

        value = Q[(s,a)]

        if best_value is None or value > best_value :
            best_value = value
            best_action = a

    return best_action , best_value

def play_episode(env,Q,WS=[1,1]):

    try:
       env.setWeights(WS)
    except Exception as e:
      pass # el entorno no tiene setWeights (no pasa nada )

    total_reward = 0.0
    state , _ = env.reset()
    state = tuple(state)

    while True:
        action , _ = best_action(env,Q,state)
        new_state, reward, is_done, tr , _ = env.step(action)
        new_state = tuple(new_state)
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

def Q_learning(env,WS,gamma=0.7,epsilon=0.5,alfa=0.7,iterations=100000,test_episodes=25):

    Q = collections.defaultdict(float) # diccionario normal solo que si no esta la clave devuelve 0 siempre
    info = []
    best_v0 = float('-inf')
    best_Q = None

    try:
       env.setWeights(WS)
    except Exception as e:
      pass # el entorno no tiene setWeights

    s0 , _ = env.reset()
    s = tuple(s0)

    for i in range(iterations):

        if random.random() > epsilon :
            a , _ = best_action(env,Q,s)
        else :
            a = env.action_space.sample()

        sn , r  , done , tr , _ = env.step(a)

        sn = tuple(sn)

        _ , value = best_action(env,Q,sn)

        Q[(s,a)] = Q[(s,a)] + alfa * ( r + gamma * value - Q[(s,a)] )

        s = sn

        if done or tr :

            v0 = V0(env,Q,epi=test_episodes,WS=WS)
            info.append(v0)

            if v0 > best_v0 or best_Q == None :
                best_v0 = v0
                best_Q = Q.copy()
                print("* Best renward:",best_v0,"iter:",i)

            s , _ = env.reset()
            s = tuple(s)

    return best_Q , info

def get_pi(env,Q):

    pi = {}

    for st , _ in Q.keys():
       a , _ = best_action(env,Q,st)
       pi[st] = a

    return pi
