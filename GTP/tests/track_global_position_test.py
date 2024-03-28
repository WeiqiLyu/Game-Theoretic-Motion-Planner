import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\\Users\\xshys\\Desktop\\Game_Theoretic_Planner\\GTP')
from utils import racing_env
from utils.constants import *

def track_global_position():
    track_spec = np.genfromtxt("data/track_layout/" + "oval_shape" + ".csv", delimiter=",")
    track = racing_env.ClosedTrack(track_spec, track_width=1.0)
    fig, ax = plt.subplots(2)
    track.plot_track(ax[0])
    num_sampling_per_meter = 100
    num_track_points = int(np.floor(num_sampling_per_meter * track.lap_length))
    points_center = np.zeros((num_track_points, 2))
    track_xglob = np.zeros((num_track_points, 4))
 
    for i in range(0, num_track_points):
        points_center[i, :] = track.get_global_position( 
            i / float(num_sampling_per_meter),
            0.0,  #ey=0
        )
        psi = track.get_orientation(i / float(num_sampling_per_meter),0)   
        curv = track.get_curvature(i / float(num_sampling_per_meter))
        track_xglob[i,0] = points_center[i, 0]
        track_xglob[i,1] = points_center[i, 1]
        track_xglob[i,2] = psi
        track_xglob[i,3] = curv

    np.savetxt("data/track/oval_shape.csv", track_xglob, delimiter=" ")
    ax[1].plot(track_xglob[:,0], track_xglob[:,1], linewidth=1)
    print(track.lap_length)
    plt.show()

if __name__ == "__main__":
    track_global_position()