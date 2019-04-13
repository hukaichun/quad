import numpy as np

from .monitor import ParticleMonitor
from .quadrotor_swarm import QuadrotorSwarm



class Quadrotors(ParticleMonitor, QuadrotorSwarm):
    def __init__(self,
        **config
    ):
        super().__init__(**config)

        self._random_position_radius = 2
        self._random_velocity_radius = 2
        self._random_angular_velocity_radius = 6



    def step(self,force):
        self.apply_force(force)
        self.eval
        return self.attitude

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
        self.attitude = (np.random.random(self.attitude.shape)-.5)*2
        self.position *= self._random_position_radius
        self.angular_velocity *= self._random_angular_velocity_radius
        self.velocity *= self._random_velocity_radius

        self.quaternion /= np.linalg.norm(self.quaternion, axis=1, keepdims=True)



    def rander(self):
        self.show()

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



