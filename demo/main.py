import quad.gym.hover_2019 as ENV
import tensorflow as tf

saver = tf.train.import_meta_graph("newquad/FRPO_newquad_3232_default_armlength0129_COUT100000.meta")
graph = tf.get_default_graph()

obs_tf = graph.get_tensor_by_name("env/obs0:0")
act_tf = graph.get_tensor_by_name("Agent/Actor/Sin:0")


config = {"length": 0.129,
          "inertia":[0.00297, 0.00333, 0.005143],
          "mass": 0.762,
          "init_num":1
          }


quad = ENV.Quadrotors(**config)




with tf.Session() as sess:
	writer = tf.summary.FileWriter("/tmp/newquad",sess.graph)
	saver.restore(sess, "newquad/FRPO_newquad_3232_default_armlength0129_COUT100000")

	obs = quad.reset()
	for _ in range(300):
		act = sess.run(act_tf, {obs_tf:obs})
		obs, _, _, _ = quad.step(act)
		quad.show()