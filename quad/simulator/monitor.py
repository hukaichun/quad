import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm     as cm
from mpl_toolkits.mplot3d import Axes3D

from .core.core import quat2rotation_np
from .core.swarm import Swarm

class ParticleMonitor(Swarm):
    def __init__(self, model_pts=None, **kwargs):
        super().__init__(**kwargs)

        if model_pts is None:
            from .skeleton.quadrotor import four_petal
            model_pts = four_petal(4)
        
        self.skeleton = model_pts
        self.GUI_init()

    def GUI_init(self, name="Quad_Sim"):
        self.num = self.attitude.shape[0]
        axes_limit = [-3,3]

        self.fig = plt.figure(name)

        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlim3d(axes_limit)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylim3d(axes_limit)
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlim3d(axes_limit)
        self.ax.set_zlabel('Z (m)')
        self.colors = cm.rainbow(np.linspace(0,1,self.num))
        self.axs = [self.ax.plot([],[],[],
                                   color=c,
                                   linewidth=1,
                                   antialiased=False)[0] 
                     for c in self.colors]

        plt.tight_layout()



    def show(self):
        rotation_matrix = quat2rotation_np(self.quaternion)
        position = self.position
        pts = np.matmul(self.skeleton, rotation_matrix)
        pts += np.expand_dims(position,1)

        for idx, ax in zip(range(self.num), self.axs):
            ax.set_data(pts[idx,:,0], pts[idx,:,1])
            ax.set_3d_properties(-pts[idx,:,2])

        plt.pause(1./600)
