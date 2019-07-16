import tensorflow as tf
import numpy as np



class Swarm:
    def __init__(self, init_num):
        with tf.name_scope("ELEMENTS"):
            init_value = np.zeros((init_num, 4)).astype("float32")
            init_value[:,0] = 1
            self._quaternion = tf.Variable(init_value, trainable=False, name="quaternion")
            self.reset_quaternion = tf.assign(self._quaternion, init_value,name="reset_quaternion")

            init_value = np.zeros((init_num,3)).astype("float32")
            self._angular_velocity = tf.Variable(init_value, trainable=False, name="angular_velocity")
            self.reset_angular_velocity = tf.assign(self._angular_velocity, init_value, name="reset_angular_velocity")

            init_value = np.zeros((init_num,3)).astype("float32")
            self._position = tf.Variable(init_value, trainable=False, name="position")
            self.reset_position = tf.assign(self._position, init_value, name="reset_position")

            init_value = np.zeros((init_num,3)).astype("float32")
            self._velocity = tf.Variable(init_value, trainable=False, name="velocity")
            self.reset_velocity = tf.assign(self._velocity, trainable=False, init_value, name="reset_velocity")

            init_value = np.zeros((init_num,3)).astype("float32")
            self._body_torque = tf.Variable(init_value, trainable=False, name="body_torque")
            self.reset_body_torque = tf.assign(self._body_torque, init_value, name="reset_body_torque")

            init_value = np.zeros((init_num,3)).astype("float32")
            self._body_force = tf.Variable(init_value, trainable=False, name="body_force")
            self.reset_body_force = tf.assign(self._body_force, init_value, name="reset_body_force")

            init_value = np.zeros((init_num,3)).astype("float32")
            self._ext_force = tf.Variable(init_value, trainable=False, name="ext_force")
            self.reset_ext_force = tf.assign(self._ext_force, init_value, name="reset_ext_force")

        



    