import numpy as np

# quaternion -->>  rotation matrix
def quat2rotation_np(quaternion):
    w = quaternion[:,0]
    i = quaternion[:,1]
    j = quaternion[:,2]
    k = quaternion[:,3]

    l = np.shape(w)[0]

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
    
    r_v  = np.stack([r_00, r_01, r_02, r_10, r_11, r_12, r_20, r_21, r_22], axis=1)
    r    = np.reshape(r_v, (l,3,3))
    return r


# d quaternion/ dt 
def d_quaternion(angular_acceleration, quaternion):
    w = quaternion[:,0]
    i = quaternion[:,1]
    j = quaternion[:,2]
    k = quaternion[:,3]

    l = np.shape(w)[0]

    m = np.stack([-i,  w, -k,  j,
                  -j,  k,  w, -i,
                  -k, -j,  i,  w], axis = 1)
    m = np.reshape(m, (l, 3, 4)) 
    angular_acceleration_m = np.expand_dims(angular_acceleration, 1)
    dq = np.squeeze(0.5*np.matmul(angular_acceleration_m, m))

    return dq


# d angular_velocity / dt
def d_angular_velocity(torque,            
                       angular_velocity,
                       inertia,
                       inertiaInv = None):
    '''
           input     |  shape  
    -----------------|---------------
    torque           | [batch, 1, 3]
    angular_velocity | [batch, 1, 3]
    inertia          | [batch, 3, 3] or [3,3]
    inertiaInv       | [batch, 3, 3] or [3,3]

    '''
    if inertiaInv is None:
        inertiaInv = np.linalg.inv(inertia)

    tmp = torque - np.cross(angular_velocity, np.matmul(angular_velocity, inertia))
    return np.matmul(tmp, inertiaInv)


#  # d state/ dt      
#  
#          arg          |  shape
#    -------------------|----------
#      orientation      | (batch, 4)
#      position         | (batch, 3)
#      angular_velocity | (batch, 3)
#      velocity         | (batch, 3)
#      body_torque      | (batch, 3)
#      body_force       | (batch, 3)
#      external_force   | (batch, 3)
#      inertia          | (batch, 3, 3) or (3,3)
#      inertia_inv      | (batch, 3, 3) or (3,3)
#      mass             | (batch, 1, 1) or ()
def d_state(orientation, position, 
            angular_velocity, velocity,
            body_torque, body_force, external_force,
            inertia, inertiaInv, mass):


    rotation_matrix = quat2rotation_np(orientation)
    rotation_matrix_t = np.transpose(rotation_matrix,
                                     axes=(0,2,1))

    angular_velocity_m = np.expand_dims(angular_velocity, 1)
    angular_velocity_body_frame_m = np.matmul(angular_velocity_m, 
                                              rotation_matrix_t)

    torque_B = np.expand_dims(body_torque,1)
    force_B = np.expand_dims(body_force,1)


    alpha_B = d_angular_velocity(
            torque_B,
            angular_velocity_body_frame_m,
            inertia, 
            inertiaInv
        )
    alpha = np.matmul(alpha_B, rotation_matrix)
    alpha = np.squeeze(alpha)

    force_total = np.matmul(force_B,rotation_matrix)+np.expand_dims(external_force,1)
    acceleration = np.squeeze(force_total/mass)

    delta_quaternion = d_quaternion(angular_velocity, orientation)
    delta_position = velocity
    delta_angular_velocity = alpha
    delta_velocity = acceleration

    return delta_quaternion, delta_position, delta_angular_velocity, delta_velocity


if __name__ == "__main__":

    gab = [
        [1,0,0,0],
        [1,0,0,1],
        [1,0,1,0],
        [1,1,0,0]
    ]
    gaba = [
        [0,0,0],
        [0,0,1],
        [0,1,0],
        [1,0,0]
        ] 
    gab = np.asarray(gab)
    gaba = np.asarray(gaba)

    r = quat2rotation_np(gab)
    print(r)

