import numpy as np
import tensorflow as tf
from core.rigidbody_tf import RigidBody_tf




class Quadrotor_tf(RigidBody_tf):
    def __init__(self,num,
        thrust,
        length,
        drag_coeff,
        deltaT,
        **config # include inertia, mass, gravity_acc  
        ):

        self.thrust = thrust

        with tf.name_scope("Quadrotor"):

            with tf.name_scope("Physical_Property"):

                thrust_2_force_trans_matrix = self.build_X_type_transform(length, drag_coeff)
            
            with tf.name_scope("Thrust_2_Force"):

                self.body_torque, self.body_force = self.buld_thrust_2_force(self.thrust, thrust_2_force_trans_matrix)

            with tf.name_scope("OBJECT"):

                super().__init__(num,
                                 self.body_torque, self.body_force, None, 
                                 **config)

            with tf.name_scope("Evaluate"):

                self.build_Eular_evaluate(deltaT)
                


    def build_X_type_transform(self, length, drag_coeff):        
        sqrt2 = np.sqrt(2.)
        thrust_2_force = np.asarray(
            [[-length/sqrt2,  length/sqrt2,  drag_coeff, 0, 0, -1.],
             [ length/sqrt2, -length/sqrt2,  drag_coeff, 0, 0, -1.],
             [ length/sqrt2,  length/sqrt2, -drag_coeff, 0, 0, -1.],
             [-length/sqrt2, -length/sqrt2, -drag_coeff, 0, 0, -1.]]
        ).astype("float32")
        return tf.constant(thrust_2_force, name="thrust_2_force_trans_matrix")


    def buld_thrust_2_force(self, thrust, transform):
        general_force = tf.matmul(thrust, transform)
        torque, force = tf.split(general_force, 2, axis=1)
        return torque, force


    def build_Eular_evaluate(self,deltaT):
        with tf.name_scope("Eular_Method"):
            self.deltaT = tf.constant(np.float32(deltaT))
            new_q = self._quaternion + self.d_quaternion*self.deltaT
            new_w = self._angular_velocity + self.d_angular_velocity*self.deltaT
            new_p = self._position + self.d_position*self.deltaT
            new_v = self._velocity + self.d_velocity*self.deltaT

            self.new_q_norm = tf.norm(new_q, axis=1, keepdims=True)

            new_q = tf.math.l2_normalize(new_q, axis=1)

            eval_q = tf.assign(self._quaternion, new_q, use_locking=True)
            eval_w = tf.assign(self._angular_velocity, new_w, use_locking=True)
            eval_p = tf.assign(self._position, new_p, use_locking=True)
            eval_v = tf.assign(self._velocity, new_v, use_locking=True)

        # evaluate = tf.group([eval_q, eval_w, eval_p, eval_v])
        self.evaluate = tf.tuple([self._quaternion, self._angular_velocity, self._position, self._velocity],
                                  name = "evaluate_q_w_p_v",
                                  control_inputs=[eval_q, eval_w, eval_p, eval_v])






