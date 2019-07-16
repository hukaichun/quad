from core.swarm_tf import Swarm
from core.core_tf import d_state

import numpy as np
import tensorflow as tf


class Quadrotor(Swarm):
    def __init__(self,init_num,
        thrust=None,
        length = 0.129,#0.105,
        drag_coeff = 0.016,
        inertia = [0.00297, 0.00333, 0.005143],#[0.0023, 0.0025, 0.0037],
        mass = 0.76,#.667,
        gravity_acc = [0,0,9.81],
        deltaT = .01,
        **kwargs
    ):
        if thrust is None:
                    self.thrust = tf.placeholder(tf.float32, (None,4), name="thrust")
                else:
                    self.thrust = thrust


        with tf.name_scope("Quadrotor"):
            super().__init__(init_num)

            with tf.name_scope("Physical_Property"):
                inertia = np.diag(inertia).astype("float32")
                self._inertia = tf.constant(inertia, name="intetia")

                inertiaInv = np.linalg.inv(inertia).astype("float32")
                self._inertiaInv = tf.constant(inertiaInv, name="inverse_inertia")

                mass = np.float32(mass)
                self._mass = tf.constant(mass, name="mass")

                gravity = np.asarray(gravity_acc)*mass
                self._gravity = tf.constant(gravity.astype("float32"), name="gravity")

                deltaT = np.float32(deltaT)
                self._deltaT = tf.constant(deltaT, name="deltaT")

                sqrt2 = np.sqrt(2.)
                thrust_2_force = np.asarray(
                    [[-length/sqrt2,  length/sqrt2,  drag_coeff, 0, 0, -1.],
                     [ length/sqrt2, -length/sqrt2,  drag_coeff, 0, 0, -1.],
                     [ length/sqrt2,  length/sqrt2, -drag_coeff, 0, 0, -1.],
                     [-length/sqrt2, -length/sqrt2, -drag_coeff, 0, 0, -1.]]
                ).astype("float32")
                self._thrust_2_force_trans_matrix = tf.constant(thrust_2_force, name="thrust_2_force")

            with tf.name_scope("Set_Random"):
                self.target_index = tf.placeholder(tf.int32, (None,), name="target_index")
                num_tf = tf.shape(self.target_index)[0]
                def Sphereical_Random():
                    r, theta, phi = tf.split(tf.random.uniform((num_tf,3)),3, axis=1)
                    theta = np.pi * theta
                    phi   = 2*np.pi * phi
                    sin_theta = tf.sin(theta)
                    cos_theta = tf.cos(theta)
                    sin_phi   = tf.sin(phi)
                    cos_phi   = tf.cos(phi)
                    return r, sin_theta*cos_phi, sin_theta*sin_phi, cos_theta

                with tf.name_scope("Random_q"):
                    quaternion = tf.random.uniform((num_tf,4))-0.5
                    quaternion/= tf.norm(quaternion, axis=1, keepdims=True)
                    self.random_q = tf.scatter_update(self._quaternion, self.target_index, quaternion)

                with tf.name_scope("Random_w"):
                    r, x, y, z = Sphereical_Random()
                    w_max = 2
                    r = w_max*r
                    random_angular_velocity = tf.concat([r*x, r*y, r*z], axis=1)
                    self.random_w = tf.scatter_update(self._angular_velocity, self.target_index, random_angular_velocity)

                with tf.name_scope("Random_p"):
                    r, x, y, z = Sphereical_Random()
                    p_min = 0.5
                    p_max = 2
                    r = (p_max-p_min)*r + p_min
                    random_position = tf.concat([r*x, r*y, r*z], axis=1)
                    self.random_p = tf.scatter_update(self._position, self.target_index, random_position)

                with tf.name_scope("Ramdom_v"):
                    r, x, y, z = Sphereical_Random()
                    v_max=2
                    r = v_max*r
                    random_velocity = tf.concat([r*x, r*y, r*z], axis=1)
                    self.random_v = tf.scatter_update(self._velocity, self.target_index, random_velocity)

            with tf.name_scope("Apply_Thrust"):
                with tf.name_scope("broadcast_gravity"):
                    gravity = tf.broadcast_to(self._gravity, (init_num,3))

                with tf.name_scope("Thrust_2_Force"):
                    general_force = tf.matmul(self.thrust, self._thrust_2_force_trans_matrix)
                    torque, force = tf.split(general_force, 2, axis=1)
                

            with tf.name_scope("Evaluate"):
                dq, dw, dp, dv = d_state(
                        self._quaternion, self._angular_velocity, 
                        self._position, self._velocity,
                        torque, force, gravity,
                        self._inertia, self._inertiaInv, self._mass)

                with tf.name_scope("Eular_Method"):
                    eval_q = tf.assign_add(self._quaternion, dq*self._deltaT)
                    eval_w = tf.assign_add(self._angular_velocity, dw*self._deltaT)
                    eval_p = tf.assign_add(self._position, dp*self._deltaT)
                    eval_v = tf.assign_add(self._velocity, dv*self._deltaT)

                self.eval = tf.group([eval_q, eval_w, eval_p, eval_v])






