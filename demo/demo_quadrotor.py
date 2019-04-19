import numpy as np
from quad.gym.hover_2018 import Quadrotors


def quad_demo(num = 10):
    
    quads = Quadrotors(init_num=num)

    s = np.copy(quads.reset())
    for _ in range(120):
        quads.render()
        random_act = np.random.random((num,4))+8
        s_next, r, done, info = quads.step(random_act)
        if any(done):
            quads.reset(done)

    quads.bound_radius_out = 3
    for _ in range(120):
        quads.render()
        random_act = np.random.random((num,4))+8
        s_next, r, done, info = quads.step(random_act)
        if any(done):
            quads.reset(done)


        
if __name__ == "__main__":
    quad_demo()

