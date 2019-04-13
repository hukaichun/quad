from .core.swarm import Swarm
from .core.core import d_state

import numpy as np

class QuadrotorSwarm(Swarm):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.set_physical_property()

        self._random_position_radius = 2
        self._random_velocity_radius = 2
        self._random_angular_velocity_radius = 6
        
    def set_physical_property(self,
        length = 0.105,
        drag_coeff = 0.016,
        inertia = [.0026, .0026, .0039],
        mass = .617,
        gravity_acc = [0,0,9.81],
        deltaT = .01
    ):
        inertia = np.diag(inertia)
        inertia_inv = np.linalg.inv(inertia)

        sqrt2 = np.sqrt(2.)
        thrust_2_force = np.asarray(
            [[-length/sqrt2,  length/sqrt2,  drag_coeff, 0, 0, -1.],
             [ length/sqrt2, -length/sqrt2,  drag_coeff, 0, 0, -1.],
             [ length/sqrt2,  length/sqrt2, -drag_coeff, 0, 0, -1.],
             [-length/sqrt2, -length/sqrt2, -drag_coeff, 0, 0, -1.]]
        )

        self.__length = length
        self.__drag_coeff = drag_coeff
        self.__inertia = inertia
        self.__inertia_inv =  inertia_inv
        self.__mass = mass
        self.__gravity = np.asarray(gravity_acc)*mass
        self.__deltaT = deltaT
        self.__thrust_2_force_trans_matrix = thrust_2_force


    def reset(self, idxs=None):
        if idxs is None:
            self.reset_all()
            return

        self.attitude[idxs] = (np.random.random(self.attitude[idxs].shape)-.5)
        self.position[idxs] *= self._random_position_radius
        self.angular_velocity[idxs] *= self._random_angular_velocity_radius
        self.velocity[idxs] *= self._random_velocity_radius

        self.quaternion[idxs] /= np.linalg.norm(self.quaternion[idxs], axis=1, keepdims=True)


    def reset_all(self):
        self.attitude = (np.random.random(self.attitude.shape)-.5)
        self.position *= self._random_position_radius
        self.angular_velocity *= self._random_angular_velocity_radius
        self.velocity *= self._random_velocity_radius

        self.quaternion /= np.linalg.norm(self.quaternion, axis=1, keepdims=True)


    def apply_force(self, body_torque, body_force, ext_force = [0,0,0]):
        self.body_force = body_force
        self.body_torque = body_torque
        self.external_force = self.__gravity + ext_force


    def apply_thrust(self, thrust):
        general_force = self.__thrust_2_force(thrust)
        torque, force = np.split(general_force, 2, axis=1)
        self.apply_force(torque, force)


    #  thrust | (batch, 1, 4) or (1,4)
    def __thrust_2_force(self, thrust):
        return np.matmul(thrust, self.__thrust_2_force_trans_matrix)


    @property
    def random_position_radius(self):
        return self._random_position_radius
    
    @random_position_radius.setter
    def random_position_radius(self, value):
        self._random_position_radius = value

    @property
    def random_velocity_radius(self):
        return self._random_velocity_radius
    
    @random_velocity_radius.setter
    def random_velocity_radius(self, value):
        self._random_velocity_radius = value


    @property
    def random_angular_velocity_radius(self):
        return self._random_angular_velocity_radius
    
    @random_angular_velocity_radius.setter
    def random_angular_velocity_radius(self, value):
        self._random_angular_velocity_radius = value

    @property
    def eval(self):
        (quaternion,
         position,
         angular_velocity,
         velocity,
         body_torque,
         body_force,
         external_force) = self.state

        (delta_quaternion,
         delta_position,
         delta_angular_velocity,
         delta_velocity) = d_state(quaternion,position,
                                   angular_velocity, velocity,
                                   body_torque, body_force, external_force,
                                   self.__inertia, self.__inertia_inv, self.__mass)


        quaternion       += delta_quaternion*self.__deltaT
        position         += delta_position*self.__deltaT
        angular_velocity += delta_angular_velocity*self.__deltaT
        velocity         += delta_velocity*self.__deltaT

        quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)


    




if __name__ == "__main__":

    quad = QuadrotorSwarm(num=3)
    print(quad.state, '\n')

    quad.apply_force([[2,2,1,1]])
    print(quad.state, '\n')

    quat = quad.quaternion
    quat[:] = np.random.random(quat.shape)-0.5
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    print(quad.quaternion)

    # for _ in range(10):
    #     quad.eval
    #     print(quad.state, '\n')

