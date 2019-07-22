import numpy as np
from .core import d_state

class RigidBody_np:
	def __init__(self, num,

            #physical constants 
            inertia=[0.00297, 0.00333, 0.005143], 
            mass=0.76, 
            gravity_acc=[0,0,9.81],
            ):

		self.num = num
		self.creat_states_variable(num)
		self.create_physical_constants(inertia, mass, gravity_acc)
		self.build_equation_of_motion()

	def creat_states_variable(self, init_num):
		self._quaternion = np.zeros((init_num, 4))
		self._quaternion[:,0] = 1
		self._angular_velocity = np.ones((init_num,3))
		self._position = np.ones((init_num,3))
		self._velocity = np.ones((init_num,3))

	def create_physical_constants(self, inertia, mass, gravity_acc):
		self._inertia = np.diag(inertia)
		self._inertiaInv = np.linalg.inv(self._inertia)
		self._mass = mass
		self._gravity = np.broadcast_to(np.asarray(gravity_acc)*mass,(self.num,3))

	def build_equation_of_motion(self):
		pass

	def d_state(self, body_torque, body_force):
		return d_state(
				self._quaternion,
				self._angular_velocity,
				self._position,
				self._velocity,
				body_torque, body_force, self._gravity,
				self._inertia, self._inertiaInv, self._mass
			)
