import tensorflow as tf
import numpy as np

from core.rigidbody_np import RigidBody_np
from core.rigidbody_tf import RigidBody_tf

num = 3

body_torque_tf = tf.placeholder(tf.float32, (None, 3), name="torque")
body_force_tf = tf.placeholder(tf.float32, (None, 3), name="force")

config = {
    "inertia": [.5,.5,1.],
    "mass": 1.,
    "gravity_acc":[0,0,10]
}

rigidbody_np = RigidBody_np(num, **config)
rigidbody_tf = RigidBody_tf(num, body_torque_tf, body_force_tf, **config)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter("/tmp/unit_test/rigidbody")

    torq = np.random.random((num, 3)).astype("float32")*2
    forc = np.random.random((num, 3)).astype("float32")

    np_result = rigidbody_np.d_state(body_torque=torq, body_force=forc)
    tf_result = sess.run(rigidbody_tf.d_state, {body_torque_tf:torq, body_force_tf:forc})

    for i,j in zip(np_result, tf_result):
    	print(i,j)
    	print(i-j)


