import tensorflow as tf
import numpy as np
from quadrotor_swarm_tf import Quadrotor_tf
from quadrotor_swarm_np import Quadrotor_np

config = {
	"length": 0.105,
	"drag_coeff": 0.016,
	"deltaT": 0.1,

	"inertia": [0.0023, 0.0025, 0.0037],
	"mass": .667,
	"gravity_acc": [0,0,9.81]
}



thrust_tf = tf.placeholder(tf.float32, (None,4))
num=3
quad_tf = Quadrotor_tf(num, thrust_tf, **config)
quad_np = Quadrotor_np(num, **config)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
	writer = tf.summary.FileWriter("/tmp/quadrotor_tf_test", sess.graph)
	sess.run(init_op)

	print("\nphysical_params:")
	def printINFO_tf(tf_Tesor):
		tmp = sess.run(tf_Tesor)
		print("{}:\n".format(tf_Tesor.name), tmp)
	
	def printINFO_np(var):
		print("np:\n", var, "\n")

	printINFO_tf(quad_tf._gravity)
	printINFO_np(quad_np._gravity)

	printINFO_tf(quad_tf._mass)
	printINFO_np(quad_np._mass)

	printINFO_tf(quad_tf._inertia)
	printINFO_np(quad_np._inertia)

	printINFO_tf(quad_tf._inertiaInv)
	printINFO_np(quad_np._inertiaInv)

	printINFO_tf(quad_tf._quaternion)
	printINFO_np(quad_np._quaternion)

	printINFO_tf(quad_tf._angular_velocity)
	printINFO_np(quad_np._angular_velocity)

	printINFO_tf(quad_tf.deltaT)
	printINFO_np(quad_np.deltaT)

	# thrust = np.random.random((3,4))
	# thrust = np.zeros((num,4))
	thrust = np.ones((num,4))
	next_state_tf = sess.run(quad_tf.evaluate,
		{thrust_tf:thrust})

	next_state_np = quad_np.evaluate(thrust)
	for i, j in zip(next_state_np, next_state_tf):
		print(i-j)


	print("\nrandom_initial")
	print("check assign")
	quat = sess.run(quad_tf._quaternion)
	quad_np._quaternion[:,:] = quat
	print(quad_np._quaternion[:,:] - sess.run(quad_tf._quaternion))

	posi = sess.run(quad_tf._position)
	quad_np._position[:,:] = posi
	print(quad_np._position - sess.run(quad_tf._position))

	angv = sess.run(quad_tf._angular_velocity)
	quad_np._angular_velocity[:,:] = angv
	print(quad_np._angular_velocity[:,:] - sess.run(quad_tf._angular_velocity))

	velo = sess.run(quad_tf._velocity)
	quad_np._velocity[:,:] = velo
	print(quad_np._velocity[:,:] - sess.run(quad_tf._velocity))

	print("check d_state")
	torq = np.random.random((num,3))
	forc = np.random.random((num,3))
	d_state_tf = sess.run(quad_tf.d_state, {quad_tf.body_torque: torq,
		                                    quad_tf.body_force: forc})
	d_state_np = quad_np.d_state(torq, forc)
	for i,j in zip(d_state_tf, d_state_np):
		print(i-j)
	print("\n\n")



	print("check update")
	next_state_np = quad_np.evaluate(thrust)
	next_state_tf, norm = sess.run([quad_tf.evaluate, quad_tf.new_q_norm],
			{thrust_tf:thrust}
		)
	for i, j in zip(next_state_np, next_state_tf):
		print(i-j,"\n")
	print(norm)
	print(np.linalg.norm(next_state_tf[0], axis=1, keepdims=True))
	print(np.linalg.norm(quad_np._quaternion[:,:], axis=1, keepdims=True))

	print("update info")
	print(next_state_tf[0] - quat - 0.1*d_state_tf[0])
	print(d_state_tf[0])
	print(next_state_np[0] - quat - 0.1*d_state_np[0])




