import numpy as np

from quad.simulator.monitor import ParticleMonitor
from quad.simulator.quadrotor_swarm import QuadrotorSwarm
from quad.simulator.core.core import quat2rotation_np
from quad.utils import cost


def attitude2state(attitude):
    quat = attitude[:,:4]
    roto = quat2rotation_np(quat)
    roto = np.transpose(roto, axes=(0,2,1))
    roto_v = roto.reshape((-1,9))
    pos = attitude[:,4:7]*0.5
    vs  = attitude[:,7:]*0.15
    state = np.concatenate([roto_v, vs, pos],axis=1)
    return state



class Quadrotors(ParticleMonitor, QuadrotorSwarm):
    def __init__(self,
        **config
    ):
        super().__init__(**config)

        self._num = config["init_num"]
        self._quaternion_goal = np.asarray([[1,0,0,0]])
        self._position_goal = np.asarray([[0,0,0]])

        self._terminal_radius = [1, 6]


    def step(self,command):
        thrust = 6*command
        thrust += 0.25*self._QuadrotorSwarm__mass*9.81
        thrust = np.clip(thrust, 0, 8)

        self.apply_thrust(thrust)
        self.eval
        self.angular_velocity = np.clip(self.angular_velocity, -20, 20)
        self.velocity = np.clip(self.velocity, -5, 5)

        r_q = cost.angular_cost(self._quaternion_goal, self.quaternion)
        r_p = cost.position_cost(self._position_goal, self.position)
        r_a = np.linalg.norm(command, axis=1, keepdims=False)

        r = r_q + r_p + r_a 
        r*=0.002

        ouside_internal = r_p < self._terminal_radius[0]
        terminal_flag = ouside_internal

        state = attitude2state(self.attitude)

        return state, -r, terminal_flag, None


    def render(self):
        self.show()


    def reset(self, idxs = None):
        if idxs is None:
            idxs = self._num*[True]
        self.random(idxs)
        return attitude2state(self.attitude)


    @property
    def bound_radius_in(self):
        return self._terminal_radius[0]
    @bound_radius_in.setter
    def bound_radius_in(self, value):
        self._terminal_radius[0] = value


    @property
    def bound_radius_out(self):
        return self._terminal_radius[1]
    @bound_radius_out.setter
    def bound_radius_out(self, value):
        self._terminal_radius[1] = value


