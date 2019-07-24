import tensorflow as tf
import numpy as np
from .core_tf import d_state


class RigidBody_tf:
    GraphKeys = []
    '''
    Rigid bodies class

    members:
        BASIC VARIABLES
        ------------------------------------------------
        num,                    int,            (),        
        _quaternion,            tf.Variable,    (num,4),  
        _angular_velocity,      tf.Variable,    (num,3),   
        _position,              tf.Variable,    (num,3),   
        _velocity,              tf.Variable,    (num,3),   



        BASIC PHYSICAL CONSTANTS
        ------------------------------------------------
        _inertia,               tf.constant,    (3,3),
        _inertiaInv,            tf.constant,    (3,3),
        _mass,                  tf.constant,    (),
        _gravity,               tf.constant,    (num,3),



        TIME DERIVATIVE OF BASIC BARIABLES
        ------------------------------------------------
        d_quaternion,           tf.Tensor,      (num,4),
        d_angular_velocity,     tf.Tensor,      (num,3),
        d_position,             tf.Tensor,      (num,3),
        d_velocity,             tf.Tensor,      (num,3),
    '''
    def __init__(self, num,

            #external force
            torque_body, force_body, force_ext=None,

            #physical constants 
            inertia=[0.00297, 0.00333, 0.005143], 
            mass=0.76, 
            gravity_acc=[0,0,9.81]
            ):
        '''
                      input        | type                   | discription
            -----------------------+------------------------+-------------------
            num                    | int const              | # of rigid bodies
                                   |                        |
            torque_body            | float array placeholder| shape:(num,3)
            force_body             | float array placeholder| shape:(num,3)
            force_ext              | float array placeholder| shape:(num,3)
                                   |                        |
            inertia                | float array const      | shape:(3,)
            mass                   | float const            | shape:()
            gravity_acc            | float array const      | shape:(3,)
                                   |                        |
            angular_velocity_range | interval const         | shape:(2,)
            position_range         | interval const         | shape:(2,)
            velocity_range         | interval const         | shape:(2,)
        '''

        self.num = num

        with tf.name_scope("STATES"):
            self.create_states_variable(num)


        with tf.name_scope("physical_constants"):
            self.create_physical_constants(inertia, mass, gravity_acc)


        with tf.name_scope("equation_of_motion"):
            if force_ext is None:
                force_ext = self._gravity
                print("ONLY GRAVITY")
            else:
                force_ext = self._gravity+force_ext
            self.build_equation_of_motion(torque_body, force_body, force_ext)


        # with tf.name_scope("Random_states"):
        #     self.build_random_initialze(angular_velocity_range, position_range, velocity_range)
        

    def create_states_variable(self, init_num):
            collection_name = "INTERNAL_STATE"
            RigidBody_tf.GraphKeys.append(collection_name)

            init_value = np.zeros((init_num, 4)).astype("float32")
            init_value[:,0] = 1
            self._quat_tf = tf.Variable(init_value,       trainable=False)
            self._angv_tf = tf.Variable(init_value[:,1:], trainable=False)
            self._posi_tf = tf.Variable(init_value[:,1:], trainable=False)
            self._velo_tf = tf.Variable(init_value[:,1:], trainable=False)

            self._quaternion = tf.identity(self._quat_tf,name="quaternion")
            self._angular_velocity = tf.identity(self._angv_tf, name="angular_velocity")
            self._position = tf.identity(self._posi_tf, name="position")
            self._velocity = tf.identity(self._velo_tf, name="velocity")

            tf.add_to_collection(collection_name, self._quaternion)
            tf.add_to_collection(collection_name, self._angular_velocity)
            tf.add_to_collection(collection_name, self._position)
            tf.add_to_collection(collection_name, self._velocity)

    def create_physical_constants(self, inertia, mass, gravity_acc):
        inertia = np.diag(inertia).astype("float32")
        self._inertia = tf.constant(inertia, name="intetia")

        inertiaInv = np.linalg.inv(inertia).astype("float32")
        self._inertiaInv = tf.constant(inertiaInv, name="inverse_inertia")

        mass = np.float32(mass)
        self._mass = tf.constant(mass, name="mass")

        gravity = np.asarray(gravity_acc)*mass
        gravity_tf = tf.constant(gravity.astype("float32"), name="gravity")
        self._gravity = tf.broadcast_to(gravity_tf, (self.num,3), name="gravity_broadcasted")


    def build_equation_of_motion(self, torque_body, force_body, force_ext):
        self.d_state = d_state(
                    self._quaternion, self._angular_velocity, 
                    self._position, self._velocity,
                    torque_body, force_body, force_ext,
                    self._inertia, self._inertiaInv, self._mass)
        dq, dw, dp, dv = self.d_state
        self.d_quaternion = dq
        self.d_angular_velocity = dw
        self.d_position = dp
        self.d_velocity = dv

