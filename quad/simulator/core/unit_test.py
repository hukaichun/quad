import tensorflow as tf
import numpy as np

import core
import core_tf
import functools

quat_tf = tf.placeholder(tf.float32, (None,4), name="quat")
angv_tf = tf.placeholder(tf.float32, (None,3), name="angv")
posi_tf = tf.placeholder(tf.float32, (None,3), name="posi")
velo_tf = tf.placeholder(tf.float32, (None,3), name="velo")

torq_tf = tf.placeholder(tf.float32, (None,3), name="torq")
forc_tf = tf.placeholder(tf.float32, (None,3), name="forc")
extF_tf = tf.placeholder(tf.float32, (None,3), name="extF")

inertia_tf = tf.placeholder(tf.float32, (3, 3), name="inertia")
inertiaInv_tf = tf.placeholder(tf.float32, (3, 3), name="inertiaInv")
mass_tf = tf.placeholder(tf.float32, (), name="mass")



rotoM_tf  = core_tf.quat2rotation(quat_tf)
d_quat_tf = core_tf.d_quaternion(angv_tf, quat_tf)
d_angv_tf = core_tf.d_angular_velocity(torq_tf, angv_tf, inertia_tf, inertiaInv_tf)
d_stat_tf = core_tf.d_state(quat_tf, angv_tf, posi_tf, velo_tf,
							torq_tf, forc_tf, extF_tf,
							inertia_tf, inertiaInv_tf, mass_tf)






init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter("/tmp/unit_test/core")

    num = 3
    quat_np = np.random.random((num,4)).astype("float32")
    rotoM = sess.run(rotoM_tf, {quat_tf: quat_np})
    print("roto_diff:", rotoM - core.quat2rotation_np(quat_np))

    angv_np = np.random.random((num,3)).astype("float32")
    d_quat = sess.run(d_quat_tf, {angv_tf:angv_np, quat_tf:quat_np})
    print("dangv_diff:", d_quat - core.d_quaternion(angv_np, quat_np))


    torq_np = np.random.random((num,3)).astype("float32")
    inertia_np = np.diag(np.random.random((3))).astype("float32")
    inertiaInv_np = np.linalg.inv(inertia_np).astype("float32")
    d_angv = sess.run(d_angv_tf, {torq_tf:torq_np, angv_tf:angv_np, inertia_tf:inertia_np, inertiaInv_tf:inertiaInv_np})
    print("angv_diff:", d_angv-core.d_angular_velocity(torq_np, angv_np, inertia_np, inertiaInv_np))

    posi_np = np.random.random((num,3)).astype("float32")
    velo_np = np.random.random((num,3)).astype("float32")
    forc_np = np.random.random((num,3)).astype("float32")
    extF_np = np.random.random((num,3)).astype("float32")
    mass_np = np.float32(np.random.random())
    d_state = sess.run(d_stat_tf, 
    	{quat_tf: quat_np,
    	 angv_tf: angv_np,
    	 posi_tf: posi_np,
    	 velo_tf: velo_np,
    	 torq_tf: torq_np,
    	 forc_tf: forc_np,
    	 extF_tf: extF_np,
    	 inertia_tf: inertia_np,
    	 inertiaInv_tf: inertiaInv_np,
    	 mass_tf: mass_np})
    d_state_np = core.d_state(
    		quat_np,
    		angv_np,
    		posi_np,
    		velo_np,
    		torq_np,
    		forc_np,
    		extF_np,
    		inertia_np,
    		inertiaInv_np,
    		mass_np
    	)
    print("d_state_diff")
    for i,j in zip(d_state, d_state_np):
    	print(i-j)


print("ok")