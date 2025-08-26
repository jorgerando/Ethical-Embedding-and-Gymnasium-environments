import collections
import numpy as np

Q = collections.defaultdict(float)

obs = np.array([1, 2, 3])
a = 2

Q[ (obs,2) ] = 1.0
