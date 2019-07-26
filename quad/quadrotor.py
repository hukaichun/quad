from quad.simulator.core import core_tf
import numpy as np


class Quadrotor:
    def __init__(self, num,
        length,
        drag_coeff,
        inertia,
        mass,
        gravity_acc
        ):
        # state : {"quaternion": tf.Variable, 
        #          "angular_velocity": tf.Varialbe, 
        #          "position":tf.Variable, 
        #          "velocity":tf.Variable}
        with tf.name_scope("Internal_States"):
            self._state = core_tf.create_states_variable(num)
            self._quat = tf.identity(self._state["quaternion"], name="quat_proxy")
            self._angv = tf.identity(self._state["angular_velocity"], name="angv_proxy")
            self._posi = tf.identity(self._state["position"], name="posi_proxy")
            self._velo = tf.identity(self._state["velocity"], name="velo_proxy")

        with tf.name_scope("Physical_Constants"):
            self._physical_constants = core_tf.create_physical_constants(inertia, mass, gravity_acc, gravity_acc, num)
            self._physical_constants["thrust2force_matrix"] = core_tf.create_X_type_transform(length, drag_coeff)

    def build_evaluation_graph(self, thrust, deltaT):
        thrust = 4.*command + 0.25*self._physical_constants["gravity"]
        thrust = tf.clip(thrust, 0, 8)
        torq, forc = tf.matmul(thrust, self._physical_constants["thrust2force_matrix"])

        with tf.name_scope("euqation_of_motion"):
            d_state = core_tf.d_state(
                    self._quat, self._angv,
                    self._posi, self._velo,
                    torq, forc, self._physical_constants["gravity"],
                    self._physical_constants["inertia"], self._physical_constants["inertiaInv"],
                    self._physical_constants["mass"]
                )
        dq, dw, dp, dv = d_state

        with tf.name_scope("eualr_method"):
            deltaT = tf.constant(np.float32(deltaT), name="deltaT")
            new_q = self._quat + dq*deltaT
            new_w = self._angv + dw*deltaT
            new_p = self._posi + dp*deltaT
            new_v = self._velo + dv*deltaT
            new_q = tf.math.l2_normalize(new_q, axis=1)

        with tf.control_dependencies([new_q, new_w, new_p, new_v]):
            eval_q = tf.assign(self._state["quaternion"], new_q)
            eval_w = tf.assign(self._state["angular_velocity"], new_w)
            eval_p = tf.assign(self._state["position"], new_p)
            eval_v = tf.assign(self._state["velocity"], new_v)

        with tf.control_dependencies([eval_q, eval_w, eval_p, eval_v]):
            self.evaluate = tf.tuple([self._state["quaternion"], self._state["angular_velocity"], self._state["position"], self._state["velocity"]], name="evaluate_q_w_p_v")

        return self.evaluate

