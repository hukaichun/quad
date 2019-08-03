import tensorflow as tf
import numpy as np
from quad import Quadrotor_tf2
from quad import Monitor

import time


config = {
    "num": 3,
    "length": 0.129,
    "drag_coeff": 0.016,
    "inertia": [.00297,.00333,.005143],
    "mass":.76,
    "gravity_acc":[0,0,9.81],
    "deltaT":0.01
}

quad_tf = Quadrotor_tf2(**config)
print(quad_tf.variables)
monitor = Monitor(quad_tf)

quad_tf.quat = np.random.random((3,4))

for _ in range(100):
    command = np.random.random((1,4))
    quad_tf.step(command)
    monitor.show()



