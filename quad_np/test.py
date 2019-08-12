import tensorflow as tf
import numpy as np
from quadrotor import Quadrotor


import time


config = {
    "num": 3,
    "length": 0.16,
    "drag_coeff": 0.016,
    "inertia": [.1,.2,.1],
    "mass":.6,
    "gravity_acc":[0,0,9.81],
    "deltaT":0.01
}

quad_tf = Quadrotor(**config)
command = np.random.random((1, 4))
quad_tf.step(command)
for v in quad_tf.variables:
    print(v.name)




