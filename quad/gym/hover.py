import numpy as np

from quad.simulator.monitor import ParticleMonitor
from quad.simulator.quadrotor_swarm import QuadrotorSwarm
from quad.utils import cost


class Quadrotors(ParticleMonitor, QuadrotorSwarm):
    def __init__(self,
        **config
    ):
        super().__init__(**config)

        self._num = config["init_num"]
        self._quaternion_goal = np.asarray([[1,0,0,0]])
        self._position_goal = np.asarray([[0,0,0]])

        self._terminal_radius = [1, 6]


    def step(self,thrust):
        thrust *= 6
        thrust += 0.25*self._QuadrotorSwarm__mass
        thrust = np.clip(thrust, 0, 8)

        self.apply_thrust(thrust)
        self.eval

        r_q = cost.angular_cost(self._quaternion_goal, self.quaternion)
        r_p = cost.position_cost(self._position_goal, self.position)

        r = r_q + r_p
        r/=2

        ouside_internal = r_p < self._terminal_radius[0]
        ouside_external = r_p > self._terminal_radius[1]
        terminal_flag = np.logical_or(ouside_internal, ouside_external)

        return self.attitude, -r, terminal_flag, {"radius": r_p,
                                                  "ouside_external": ouside_external,
                                                  "ouside_internal": ouside_internal} 


    def render(self):
        self.show()


    def reset(self, idxs = None):
        if idxs is None:
            idxs = self._num*[True]
        self.random(idxs)
        return self.attitude


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


