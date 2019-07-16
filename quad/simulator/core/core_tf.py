import tensorflow as tf

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



# d quaternion/ dt 
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
def d_angular_velocity(torque,            
                       angular_velocity,
                       inertia,
                       inertiaInv):
    '''
           input     |  shape  
    -----------------|---------------
    torque           | [batch, 1, 3]
    angular_velocity | [batch, 1, 3]
    inertia          | [batch, 3, 3] or [3,3]
    inertiaInv       | [batch, 3, 3] or [3,3]

    '''
    with tf.name_scope("dw_dt"):
        torque_m = tf.expand_dims(torque, 1)
        angular_velocity_m = tf.expand_dims(angular_velocity, 1)

        tmp = torque_m - tf.cross(angular_velocity_m, tf.matmul(angular_velocity_m, inertia))
        d_w = tf.matmul(tmp, inertiaInv)
    return tf.reshape(d_w, (-1,3))



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






if __name__ == "__main__":
    import core
    import numpy as np

    #build tf graph

    quat_tf = tf.placeholder(tf.float32, (None, 4))
    angv_tf = tf.placeholder(tf.float32, (None, 3))
    posi_tf = tf.placeholder(tf.float32, (None, 3))
    velo_tf = tf.placeholder(tf.float32, (None, 3))
    torq_tf = tf.placeholder(tf.float32, (None, 3))
    forc_tf = tf.placeholder(tf.float32, (None, 3))
    extF_tf = tf.placeholder(tf.float32, (None, 3))
    inertia_tf = tf.placeholder(tf.float32, (3, 3))
    inertiaInv_tf = tf.placeholder(tf.float32, (3, 3))
    mass_tf = tf.placeholder(tf.float32, ())
    

    rotationMatrix = quat2rotation(quat_tf)
    d_q = d_quaternion(angv_tf, quat_tf)
    d_a = d_angular_velocity(torq_tf, angv_tf, inertia_tf, inertiaInv_tf)
    d_s = d_state(quat_tf, angv_tf,
                  posi_tf, velo_tf,
                  torq_tf, forc_tf, extF_tf,
                  inertia_tf, inertiaInv_tf, mass_tf)




    with tf.Session() as sess:
        num = 2
        quat = np.random.random((num,4)).astype("float32")
        angv = np.random.random((num,3)).astype("float32")
        posi = np.random.random((num,3)).astype("float32")
        velo = np.random.random((num,3)).astype("float32")
        torq = np.random.random((num,3)).astype("float32")
        forc = np.random.random((num,3)).astype("float32")
        extF = np.random.random((num,3)).astype("float32")
        inertia = np.random.random((3,3)).astype("float32")
        inertiaInv = np.linalg.inv(inertia).astype("float32")
        mass = np.float32(np.random.random())


        





        roto_tf = sess.run(rotationMatrix, {quat_tf: quat})
        print("rotationMatrix:", core.quat2rotation_np(quat)-roto_tf)

        d_quat = sess.run(d_q, {quat_tf: quat, angv_tf: angv})
        print("dq:", core.d_quaternion(angv, quat) - d_quat)

        d_angv = sess.run(d_a, {torq_tf: torq, angv_tf: angv, inertia_tf: inertia, inertiaInv_tf: inertiaInv})
        print("d_angv:",core.d_angular_velocity(
                np.expand_dims(torq,1), 
                np.expand_dims(angv,1),
                inertia,
                inertiaInv) - d_angv)

        d_stat = sess.run(d_s, {quat_tf:quat, angv_tf:angv, posi_tf:posi, velo_tf:velo, 
                                torq_tf: torq, forc_tf:forc, extF_tf:extF, 
                                inertia_tf:inertia, inertiaInv_tf:inertiaInv, mass_tf:mass})
        d_stat_np = core.d_state(quat, posi, angv, velo, torq, forc, extF, inertia, inertiaInv, mass)
        print("d_state")
        dq, da, dp, dv = d_stat
        q, p, a ,v = d_stat_np
        print(dq-q)
        print(da-a)
        print(dp-p)
        print(dv-v)


    print("ok")
