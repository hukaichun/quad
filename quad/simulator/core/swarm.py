import numpy as np


class _info:
    def __init__(self,init_num, dim):
        self.__info = np.zeros((init_num, dim))
        self.__dim = dim

    def __getitem__(self, keys):
        return self.__info[keys]

    def __setitem__(self, keys, values):
        self.__info[keys] = values

    def expand(self, num=1):
        self.__info = np.concatenate([self.__info, np.zeros((num, self.__dim))])

    def delete(self, idx=0):
        self.__info = np.delete(self.__info, idx, 0)

    def split(self, indices_or_sections):
        return np.split(self.__info, indices_or_sections, axis=1)




class Swarm(_info):
    def __init__(self,
        init_num=1
    ):
        '''
            state:
                quaternion(4), position(3), angular_velocity(3), velocity(3)
            
            force:
                body_torque(3), body_force(3), external_force(3)

        '''
        super().__init__(init_num, 4+3+3+3+3+3+3)
        self[:,0]=1

    @property
    def state(self):
        return self.split([4,7,10,13,16,19])

    @state.setter
    def state(self,value):
        self[:] = value
        return value


    @property
    def attitude(self):
        return self[:,:13]

    @attitude.setter
    def attitude(self, value):
        self[:,:13] = value
        return value
    

    @property
    def quaternion(self):
        return self[:,:4]

    @quaternion.setter
    def quaternion(self, value):
        self[:,:4] = value
        return value


    @property
    def position(self):
        return self[:,4:7]

    @position.setter
    def position(self, value):
        self[:,4:7] = value
        return value


    @property
    def angular_velocity(self):
        return self[:,7:10]

    @angular_velocity.setter
    def angular_velocity(self,value):
        self[:,7:10] = value
        return value


    @property
    def velocity(self):
        return self[:,10:13]

    @velocity.setter
    def velocity(self, value):
        self[:,10:13] = value
        return value


    @property
    def body_torque(self):
        return self[:,13:16]

    @body_torque.setter
    def body_torque(self,value):
        self[:,13:16] = value


    @property
    def body_force(self):
        return self[:,16:19]

    @body_force.setter
    def body_force(self,value):
        self[:,16:19] = value
        return value


    @property
    def external_force(self):
        return self[:,19:]
    
    @external_force.setter
    def external_force(self,value):
        self[:,19:] = value
        return value
    
    
    

     





if __name__ == "__main__":
    obj = Swarm(2)
    for _ in obj.state:
        print(_)
