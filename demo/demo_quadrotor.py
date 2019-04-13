
from quad.simulator.quadrotor import Quadrotors
from quad.utils import cost

import numpy as np

num = 10
quads = Quadrotors(init_num=num)
quads.reset()



goal_q = np.asarray([[1,0,0,0]])
goal_p = np.asarray([[0,0,0]])


for _ in range(10):
    for __ in range(100):
        quads.rander()
        states = quads.step([[1.6,1.6,1.5,1.5]])
        print(states)
        quads.reset([0,1,2])
    quads.reset()
