import tensorflow as tf
import numpy as np
from quad.simulator.quadrotor_swarm_tf import Quadrotor_tf
from quad.simulator.quadrotor_swarm_np import Quadrotor_np


endblock = "-----------------------"


config = {
    "length": 0.105,
    "drag_coeff": 0.016,
    "deltaT": 1,

    "inertia": [0.0023, 0.0025, 0.0037],
    "mass": .667,
    "gravity_acc": [0,0,9.81]
}



thrust_tf = tf.placeholder(tf.float32, (None,4))
num=4
quad_tf = Quadrotor_tf(num, thrust_tf, **config)
quad_np = Quadrotor_np(num, **config)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("/tmp/quadrotor_tf_test", sess.graph)
    sess.run(init_op)

    
    def printINFO_tf(tf_Tesor):
        tmp = sess.run(tf_Tesor)
        print("{}:\n".format(tf_Tesor.name), tmp)
    
    def printINFO_np(var):
        print("np:\n", var, "\n")
    print("\nphysical_params:")
    printINFO_tf(quad_tf._gravity)
    printINFO_np(quad_np._gravity)

    printINFO_tf(quad_tf._mass)
    printINFO_np(quad_np._mass)

    printINFO_tf(quad_tf._inertia)
    printINFO_np(quad_np._inertia)

    printINFO_tf(quad_tf._inertiaInv)
    printINFO_np(quad_np._inertiaInv)

    printINFO_tf(quad_tf._quaternion)
    printINFO_np(quad_np._quaternion)

    printINFO_tf(quad_tf._angular_velocity)
    printINFO_np(quad_np._angular_velocity)

    printINFO_tf(quad_tf.deltaT)
    printINFO_np(quad_np.deltaT)

    thrust = np.random.random((num,4))
    next_state_tf = sess.run(quad_tf.evaluate,
        {thrust_tf:thrust})

    next_state_np = quad_np.evaluate(thrust)
    for i, j in zip(next_state_np, next_state_tf):
        print(i-j)

    next_state_tf = sess.run(quad_tf.evaluate,
        {thrust_tf:thrust})

    next_state_np = quad_np.evaluate(thrust)
    for i, j in zip(next_state_np, next_state_tf):
        print(i-j)
    print(endblock)


    print("\nrandom_initial")
    print("check assign")
    quat = sess.run(quad_tf._quaternion)
    quad_np._quaternion[:,:] = quat
    print(quad_np._quaternion[:,:] - sess.run(quad_tf._quaternion))

    posi = sess.run(quad_tf._position)
    quad_np._position[:,:] = posi
    print(quad_np._position - sess.run(quad_tf._position))

    angv = sess.run(quad_tf._angular_velocity)
    quad_np._angular_velocity[:,:] = angv
    print(quad_np._angular_velocity[:,:] - sess.run(quad_tf._angular_velocity))

    velo = sess.run(quad_tf._velocity)
    quad_np._velocity[:,:] = velo
    print(quad_np._velocity[:,:] - sess.run(quad_tf._velocity))
    print(endblock)

    print("compare force")
    torq_np, forc_np = quad_np.thrust_2_force_trans(thrust, quad_np.thrust_2_force_trans_matrix)
    torq_tf, forc_tf = sess.run([quad_tf.body_torque, quad_tf.body_force],
        {
            quad_tf.thrust:thrust
        })
    print(torq_np-torq_tf)
    print(forc_np-forc_tf)
    print(endblock)


    print("check d_state")
    d_state_tf = sess.run([quad_tf.d_quaternion,
                           quad_tf.d_angular_velocity,
                           quad_tf.d_position,
                           quad_tf.d_velocity], {quad_tf.thrust:thrust})
    torq_np, forc_np = quad_np.thrust_2_force_trans(thrust, quad_np.thrust_2_force_trans_matrix)
    d_state_np = quad_np.d_state(torq_np, forc_np)
    for i,j in zip(d_state_tf, d_state_np):
        print(i-j)
        print("\n")
    print(endblock)


    print("check eular")
    q_tf = sess.run(quad_tf._quaternion)
    dt_tf = sess.run(quad_tf.deltaT)
    dq_tf = sess.run(quad_tf.d_quaternion, {quad_tf.thrust:thrust})
    newq_tf = sess.run(quad_tf.new_q, {quad_tf.thrust:thrust})
    print(q_tf+dt_tf*dq_tf-newq_tf)

    q_np = quad_np._quaternion
    dt_np = quad_np.deltaT
    dq_np = quad_np.d_state(torq_np, forc_np)[0]
    print(q_np-q_tf)
    print(dt_tf-dt_np)
    print(dq_tf-dq_np)
    print(endblock)


    print("check update")
    next_state_np = quad_np.evaluate(thrust)
    next_state_tf = sess.run(quad_tf.evaluate,
            {thrust_tf:thrust}
        )
    for i, j in zip(next_state_np, next_state_tf):
        print(i-j,"\n")
    print(endblock)


    print("check assign to tf")
    q = np.random.random((num,4))
    q/= np.linalg.norm(q, axis=1, keepdims=True)
    w = np.random.random((num,3))
    p = np.random.random((num,3))
    v = np.random.random((num,3))
    t = np.random.random((num,4))

    quad_np._quaternion[:,:] = q
    quad_np._angular_velocity[:,:] = w
    quad_np._position[:,:] = p
    quad_np._velocity[:,:] = v

    next_state_np = quad_np.evaluate(t)
    next_state_tf = sess.run(quad_tf.evaluate,
            {quad_tf._quaternion: q,
             quad_tf._angular_velocity: w,
             quad_tf._position: p,
             quad_tf._velocity: v,
             quad_tf.thrust: t}
        )
    for i, j in zip(next_state_np, next_state_tf):
        print(i-j,"\n")

    next_state_np = quad_np.evaluate(t)
    next_state_tf = sess.run(quad_tf.evaluate,
            {quad_tf.thrust: t}
        )
    for i, j in zip(next_state_np, next_state_tf):
        print(i-j,"\n")

    # internal_state = sess.run([quad_tf._quaternion, quad_tf._angular_velocity, quad_tf._position, quad_tf._velocity])
    # for i, j in zip(next_state_np, internal_state):
    #     print(i-j, "\n")












