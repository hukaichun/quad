import tensorflow as tf
from . import util
import numpy as np


class Quadrotor_tf2(tf.Module):
    def __init__(self, 
            num, 
            length, 
            drag_coeff, 
            inertia, 
            mass, 
            gravity_acc, 
            deltaT, 
            name="Quadrotor"):
        super(Quadrotor_tf2, self).__init__(name)
        with self.name_scope:
            with tf.name_scope("STATE"):
                (self._quat, 
                 self._angv, 
                 self._posi, 
                 self._velo) = util.create_states_variable(num)

            with tf.name_scope("CONFIG"):
                (self._inertia,
                 self._inertiaInv,
                 self._mass,
                 self._gravity) = util.create_physical_constants(inertia, mass, gravity_acc)

                self._thrust2force_matrix = util.create_X_type_transform(length, drag_coeff)
                self._deltaT = tf.Variable(deltaT, trainable=False, name="deltaT")
                self._num    = tf.Variable(num, trainable=False, name="number_of_quad")



    @tf.function(input_signature=(
            tf.TensorSpec(shape=(None, 4)),
            tf.TensorSpec(shape=(), dtype=tf.bool)
        ))
    def step(self, command, update=True):
        with self.name_scope:
            torq, forc = self.command2force(command)
            with tf.name_scope("Eular_Intergal"):
                dq, dw, dp, dv = self.d_state(torq, forc)
                tmp_q = self._quat + self._deltaT*dq
                tmp_w = self._angv + self._deltaT*dw
                tmp_p = self._posi + self._deltaT*dp
                tmp_v = self._velo + self._deltaT*dv

            with tf.name_scope("Clip_Boundary"):
                new_q = tf.math.l2_normalize(tmp_q, axis=1)
                new_w = tf.clip_by_value(tmp_w, -20, 20)
                new_p = tf.clip_by_value(tmp_p, -10, 10)
                new_v = tf.clip_by_value(tmp_v, -5, 5)

            if update:
                self._quat.assign(new_q)
                self._angv.assign(new_w)
                self._posi.assign(new_p)
                self._velo.assign(new_v)

            return new_q, new_w, new_p, new_v


    def d_state(self, torq, forc):
        with self.name_scope:
            extF = tf.broadcast_to(self._gravity, (self._num, 3))

            dq, dw, dp, dv = util.d_state(
                    self._quat, self._angv,
                    self._posi, self._velo,
                    torq, forc, extF,
                    self._inertia, self._inertiaInv, self._mass
                )
        return dq, dw, dp, dv


    def command2force(self, command):
        new_command = 4.*command+0.25*self._mass*9.81
        new_command = tf.clip_by_value(new_command, 0, 8)
        general_force = tf.matmul(new_command, self._thrust2force_matrix)
        torq, forc = tf.split(general_force, 2, axis=1)
        return torq, forc

    @property
    def quat(self):
        return self._quat.numpy()
    @quat.setter
    def quat(self, value):
        self._quat.assign(value)

    @property
    def angv(self):
        return self._angv.numpy()
    @angv.setter
    def angv(self, value):
        self._angv.assign(value)

    @property
    def posi(self):
        return self._posi.numpy()
    @posi.setter
    def posi(self, value):
        self._posi.assign(value)

    @property
    def velo(self):
        return self._velo.numpy()
    @velo.setter
    def velo(self,value):
        self._velo.assign(value)
    
    

