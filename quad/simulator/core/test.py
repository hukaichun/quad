import tensorflow as tf
import core_tf

import numpy as np

num = 5
quat_np = np.random.random((num,4))
angv_np = np.random.random((num,3))
posi_np = np.random.random((num,3))
velo_np = np.random.random((num,3))
torq_np = np.random.random((num,3))
forc_np = np.random.random((num,3))
extF_np = np.random.random((num,3))
inertia = np.diag(np.random.random(3))
inertiaInv = np.linalg.inv(inertia)
mass = np.random.random()



rotationMatrix   = core_tf.quat2rotation(quat_np)
Dquaternion      = core_tf.d_quaternion(angv_np, quat_np)
DangularVelocity = core_tf.d_angular_velocity(torq_np, angv_np, inertia, inertiaInv)
Dstate           = core_tf.d_state(quat_np, angv_np, posi_np, velo_np, torq_np, forc_np, extF_np, inertia, inertiaInv, mass)



print(rotationMatrix)
print(Dquaternion)
print(DangularVelocity)
print(Dstate)


transM = core_tf.create_X_type_transform(1,1)
print(transM)