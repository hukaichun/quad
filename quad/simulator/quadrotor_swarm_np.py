import numpy as np
import functools
from .core.rigidbody_np import RigidBody_np



class Quadrotor_np(RigidBody_np):
    def __init__(self, num,
        length,
        drag_coeff,
        deltaT,
        **config
        ):
        super().__init__(num,**config)
        self.thrust_2_force_trans_matrix = self.build_X_type_transform(length, drag_coeff)
        self.build_Eular_evaluate(deltaT)

    def build_X_type_transform(self, length, drag_coeff):
        sqrt2 = np.sqrt(2.)
        thrust_2_force = np.asarray(
            [[-length/sqrt2,  length/sqrt2,  drag_coeff, 0, 0, -1.],
             [ length/sqrt2, -length/sqrt2,  drag_coeff, 0, 0, -1.],
             [ length/sqrt2,  length/sqrt2, -drag_coeff, 0, 0, -1.],
             [-length/sqrt2, -length/sqrt2, -drag_coeff, 0, 0, -1.]]
        ).astype("float32")
        return thrust_2_force

    def thrust_2_force_trans(self, thrust, transform):
        general_force = np.matmul(thrust, transform)
        torque, force = np.split(general_force, 2, axis=1)
        return torque, force

    def build_Eular_evaluate(self, deltaT):
        self.deltaT = deltaT

    def evaluate(self, thrust):
        torque, force = self.thrust_2_force_trans(thrust, self.thrust_2_force_trans_matrix)
        d_q, d_w, d_p, d_v = self.d_state(body_torque=torque, body_force=force)

        self._quaternion+=d_q*self.deltaT
        self._angular_velocity+=d_w*self.deltaT
        self._position+=d_p*self.deltaT
        self._velocity+=d_v*self.deltaT

        self._quaternion/=np.linalg.norm(self._quaternion, axis=1, keepdims=True)

        return self._quaternion, self._angular_velocity, self._position, self._velocity

