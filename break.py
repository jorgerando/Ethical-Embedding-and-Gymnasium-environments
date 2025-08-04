import sys
sys.path.insert(0, 'algorithms/.')
from Q_learning import *
from OLS_Embding import *

from gymnasium_envs.BreakableBottles  import *

w = 26.1

if __name__ == '__main__':

    env = GymBreakableBottles()
    Q = Q_learning(env,[1,w], iterations=1000000 )
    pi = get_pi(env,Q)

    st , _ = env.reset()
    actions = []
    os.system('clear')

    while True:
      env.render()
      time.sleep(1)
      os.system('clear')

      at = pi[st]
      str_a = env.action2string[int(at)]
      actions.append(str_a)
      st_n, r, done ,  _ , info = env.step(at)
      st = st_n

      if done :
        st , _ = env.reset()
        print(actions)
        break
