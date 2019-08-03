import numpy as np

def four_petal(num_pt):
    num_pt = 4*num_pt+1
    theta = np.linspace(0,2*np.pi, num_pt)
    r     = 0.15*(1-np.cos(4*theta))
    pts = np.zeros((num_pt+1,3))
    pts[:-1,0] = r*np.cos(theta)
    pts[:-1,1] = r*np.sin(theta)
    pts[-1,-1] = -.1

    return pts