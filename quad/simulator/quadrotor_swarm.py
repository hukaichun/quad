from .core.swarm import Swarm
from .core.core import d_state

import numpy as np

class QuadrotorSwarm(Swarm):
    def __init__(self,init_num,
        length = 0.105,
        drag_coeff = 0.016,
        inertia = [0.0023, 0.0025, 0.0037],
        mass = .667,
        gravity_acc = [0,0,9.81],
        deltaT = .01,
        **kwargs
    ):
        super().__init__(init_num)
        
        self.set_physical_property(
            length = length,
            drag_coeff = drag_coeff,
            inertia = inertia,
            mass = mass,
            gravity_acc = gravity_acc,
            deltaT = deltaT)

        self._random_position_radius = [0.1,2]
        self._random_velocity_radius = 2
        self._random_angular_velocity_radius = 6
        
    def set_physical_property(self,
        length,
        drag_coeff,
        inertia,
        mass,
        gravity_acc,
        deltaT
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


    def random(self, idxs):
        self.random_q(idxs)
        self.random_p(idxs, *self._random_position_radius)
        self.random_v(idxs)
        self.random_w(idxs)


    def random_q(self, idxs):
        self.quaternion[idxs] = np.random.random(self.quaternion[idxs].shape)-0.5
        self.quaternion[idxs] /= np.linalg.norm(self.quaternion[idxs], axis=1, keepdims=True)

    def random_p(self, idxs, r_min=2, r_max=3):
        sphereical_coord = np.random.random(self.position[idxs].shape)
        sphereical_coord *= np.asarray([[r_max-r_min, np.pi, 2*np.pi]]) 
        sphereical_coord[:,0] += r_min

        r = sphereical_coord[:,0]
        sin_theta = np.sin(sphereical_coord[:,1])
        cos_theta = np.cos(sphereical_coord[:,1])
        sin_phi   = np.sin(sphereical_coord[:,2])
        cos_phi   = np.cos(sphereical_coord[:,2])

        self.position[idxs,0] = r*sin_theta*cos_phi
        self.position[idxs,1] = r*sin_theta*sin_phi
        self.position[idxs,2] = r*cos_theta 

    def random_v(self, idxs):
        self.velocity[idxs] = np.random.random(self.velocity[idxs].shape)-0.5
        self.velocity[idxs] *= self._random_velocity_radius*2 

    def random_w(self,idxs):
        self.angular_velocity[idxs] = np.random.random(self.angular_velocity[idxs].shape)-0.5
        self.angular_velocity[idxs] *= 2*self.random_angular_velocity_radius


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

