import tensorflow as tf

@tf.function(input_signature=(tf.TensorSpec(shape=[None, 4], dtype=tf.float32),))
def quat2rotation(quaternion):
    with tf.name_scope("Quaternion2RotationMatrix"):
        w = quaternion[:,0]
        i = quaternion[:,1]
        j = quaternion[:,2]
        k = quaternion[:,3]

        ii2 = 2 * i * i
        jj2 = 2 * j * j
        kk2 = 2 * k * k
        ij2 = 2 * i * j
        wk2 = 2 * w * k
        ki2 = 2 * k * i
        wj2 = 2 * w * j
        jk2 = 2 * j * k
        wi2 = 2 * w * i

        r_00 = 1. - jj2 - kk2
        r_10 = ij2 - wk2
        r_20 = ki2 + wj2
        r_01 = ij2 + wk2
        r_11 = 1. - ii2 - kk2
        r_21 = jk2 - wi2
        r_02 = ki2 - wj2
        r_12 = jk2 + wi2
        r_22 = 1. - ii2 - jj2
        
        r_v  = tf.stack([r_00, r_01, r_02, r_10, r_11, r_12, r_20, r_21, r_22], axis=1)
        rotation_matrix    = tf.reshape(r_v, (-1,3,3))
    return rotation_matrix



@tf.function(input_signature=[tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, 4], dtype=tf.float32)])
def d_quaternion(angular_velocity, quaternion):
    with tf.name_scope("dq_dt"):
        w = quaternion[:,0]
        i = quaternion[:,1]
        j = quaternion[:,2]
        k = quaternion[:,3]

        m = tf.stack([-i,  w,  k, -j,
                      -j, -k,  w,  i,
                      -k,  j, -i,  w], axis=1)
        m = tf.reshape(m, (-1, 3, 4)) 
        angular_velocity_m = tf.expand_dims(angular_velocity, 1)
        dq = 0.5*tf.matmul(angular_velocity_m, m)
    return tf.reshape(dq, (-1,4))



# d angular_velocity / dt
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
                              tf.TensorSpec(shape=[3,3], dtype=tf.float32),
                              tf.TensorSpec(shape=[3,3], dtype=tf.float32)])
def d_angular_velocity(torque,            
                       angular_velocity,
                       inertia,
                       inertiaInv):
    '''
           input     |  shape  
    -----------------|---------------
    torque           | [batch, 3]
    angular_velocity | [batch, 3]
    inertia          | [3,3]
    inertiaInv       | [3,3]

    '''
    with tf.name_scope("dw_dt"):
        torque_m = tf.expand_dims(torque, 1)
        angular_velocity_m = tf.expand_dims(angular_velocity, 1)

        tmp = torque_m - tf.linalg.cross(angular_velocity_m, tf.matmul(angular_velocity_m, inertia))
        d_w = tf.matmul(tmp, inertiaInv)
    return tf.reshape(d_w, (-1,3))


@tf.function(input_signature=[tf.TensorSpec(shape=[None,4], dtype=tf.float32),
                              tf.TensorSpec(shape=[None,3], dtype=tf.float32),
                              tf.TensorSpec(shape=[None,3], dtype=tf.float32),
                              tf.TensorSpec(shape=[None,3], dtype=tf.float32),
                              tf.TensorSpec(shape=[None,3], dtype=tf.float32),
                              tf.TensorSpec(shape=[None,3], dtype=tf.float32),
                              tf.TensorSpec(shape=[None,3], dtype=tf.float32),
                              tf.TensorSpec(shape=[3,3], dtype=tf.float32),
                              tf.TensorSpec(shape=[3,3], dtype=tf.float32),
                              tf.TensorSpec(shape=(), dtype=tf.float32)])
def d_state(orientation, angular_velocity,
            position, velocity,
            body_torque, body_force, external_force,
            inertia, inertiaInv, mass):
    '''     
     
            arg           |     shape             |  remark
       -------------------|-----------------------|-----------
         orientation      | (batch, 4)            |
         position         | (batch, 3)            |  
         angular_velocity | (batch, 3)            |  body frame
         velocity         | (batch, 3)            |  inertia frame
         body_torque      | (batch, 3)            |  body frame
         body_force       | (batch, 3)            |  body frame
         external_force   | (batch, 3)            |  inertia frame
         inertia          | (batch, 3, 3) or (3,3)|  
         inertiaInv       | (batch, 3, 3) or (3,3)|
         mass             | (batch, 1, 1) or ()   |

            return        |     shape              
       -------------------|-----------------------
         d_q              | (batch, 4)
         d_a              | (batch, 3)
         d_p              | (batch, 3)
         d_v              | (batch, 3)
    '''
    with tf.name_scope("ds_dt"):
        rotation_matrix = quat2rotation(orientation)
        # rotation_matrix_t = tf.transpose(rotation_matrix, perm=(0,2,1))

        force_B = tf.expand_dims(body_force, 1)
        force_total = tf.matmul(force_B, rotation_matrix)+tf.expand_dims(external_force, 1)
        acceleration = tf.reshape(force_total/mass, (-1,3))

        dq_dt = d_quaternion(angular_velocity, orientation)
        dw_dt = d_angular_velocity(body_torque,angular_velocity,inertia,inertiaInv)
        dp_dt = tf.identity(velocity)
        dv_dt = acceleration

    return dq_dt, dw_dt, dp_dt, dv_dt


def create_X_type_transform(length, drag_coeff):
    import numpy as np
    sqrt2 = np.sqrt(2.)
    thrust_2_force = np.asarray(
        [[-length/sqrt2,  length/sqrt2,  drag_coeff, 0, 0, -1.],
         [ length/sqrt2, -length/sqrt2,  drag_coeff, 0, 0, -1.],
         [ length/sqrt2,  length/sqrt2, -drag_coeff, 0, 0, -1.],
         [-length/sqrt2, -length/sqrt2, -drag_coeff, 0, 0, -1.]]
    ).astype("float32")
    return tf.Variable(thrust_2_force, trainable=False, name="thrust_2_force_trans_matrix")


def create_states_variable(num):
    import numpy as np
    init_value = np.zeros((num, 4)).astype("float32")
    init_value[:,0] = 1

    return (tf.Variable(init_value,       trainable=False, name="quaternion"),
            tf.Variable(init_value[:,1:], trainable=False, name="angular_velocity"),
            tf.Variable(init_value[:,1:], trainable=False, name="position"),
            tf.Variable(init_value[:,1:], trainable=False, name="velocity")) 
    

def create_physical_constants(inertia, mass, gravity_acc, num=None):
    import numpy as np
    inertia = np.diag(inertia).astype("float32")
    inertiaInv = np.linalg.inv(inertia).astype("float32")
    mass = np.float32(mass)
    gravity = np.asarray(gravity_acc)*mass
    if num is not None:
        gravity = np.broadcast_to(gravity, (num,3))

    return ( tf.Variable(inertia, trainable=False, name="intetia"),
             tf.Variable(inertiaInv, trainable=False, name="inverse_inertia"),
             tf.Variable(mass, trainable=False, name="mass"),
             tf.Variable(gravity.astype("float32"), trainable=False, name="gravity") )



    
