import numpy as np




# quat1 & quat2 must rank 2
def quatarnion_difference(quat1, quat2):
    diff = (quat1+quat2)# init

    diff[:,0] = 1-np.abs(np.sum(quat1*quat2, axis=1))
    diff[:,1:] = quat1[:,[0]]*quat2[:,1:] + quat2[:,[0]]*quat1[:,1:] + np.cross(quat1[:,1:], quat2[:,1:])
    return diff



def angular_cost(goal, current, keepdims=False):
    diff = quatarnion_difference(goal, current)
    return np.linalg.norm(diff, axis=1, keepdims=keepdims)

def position_cost(goal, current, keepdims=False):
    return np.linalg.norm(qoal-current, axis=1, keepdims=keepdims)
