
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\\Users\\xshys\\Desktop\\Game_Theoretic_Planner\\GTP')
from utils import base
from utils import offboard
from utils import  racing_env
from utils.constants import *



def se_ibr_single(system_param,i):
    track_layout = "oval_shape"  
    track_spec = np.genfromtxt("data/track_layout/" + "oval_shape" + ".csv", delimiter=",") 
    track = racing_env.ClosedTrack(track_spec, track_width=1.0)
    track_center = np.genfromtxt("data/track/" + track_layout + ".csv", delimiter=" ") # pro row:[X,Y,psi,curv]        

    timestep = 1.0 /8.0       # Euler discretization time 
    noise_x = np.maximum(0.3, np.minimum(np.random.randn()+1.3, 2.3))  # randomly starting position
    noise_y = np.maximum(-0.5, np.minimum(np.random.randn()*0.5, 0.5))  
    
    ego = set_up_ego(timestep, track, track_center,noise_x,noise_y,system_param)
    se_ibr_planner = set_up_se_ibr(timestep, track, track_center, ego.system_param)
    ego.set_ctrl_policy(se_ibr_planner)
 
    # define a simulator
    simulator = offboard.CarRacingSim()
    simulator.set_timestep(timestep)
    simulator.set_track(track,track_center)
    simulator.add_vehicle(ego) #se_ibr

    se_ibr_planner.set_racing_sim(simulator)
    se_ibr_planner.set_vehicles_track()

    simulator.sim(sim_time=30.0, two_lap=True, two_lap_name="ego") 
    a_max = ego.system_param.a_max
    v_max = ego.system_param.v_max
    c_max = ego.system_param.curv_max
    print(a_max)
    print(v_max)
    print(c_max)
    simulator.plot_simulation()
    file_name = 'media/animation/se_ibr_single/a' + str(a_max) + 'v' + str(v_max) + 'c' + str(c_max) + '_' + str(i)+ '.png' 
    plt.savefig(file_name)

def mpc_single(system_param,i):
    track_layout = "oval_shape"  
    track_spec = np.genfromtxt("data/track_layout/" + "oval_shape" + ".csv", delimiter=",") 
    track = racing_env.ClosedTrack(track_spec, track_width=1.0)
    track_center = np.genfromtxt("data/track/" + track_layout + ".csv", delimiter=" ")

    timestep = 1.0 /8.0   #Euler discretization time 
    noise_x = np.maximum(0.3, np.minimum(np.random.randn()+1.3, 2.3))  # randomly starting position
    noise_y = np.maximum(-0.5, np.minimum(np.random.randn()*0.5, 0.5)) 
    
    ego = set_up_ego(timestep, track, track_center,noise_x,noise_y,system_param)
    mpc_planner = set_up_mpc(timestep, track, track_center, ego.system_param)
    ego.set_ctrl_policy(mpc_planner)
 
    # define a simulator
    simulator = offboard.CarRacingSim()
    simulator.set_timestep(timestep)
    simulator.set_track(track,track_center)
    simulator.add_vehicle(ego) #mpc

    mpc_planner.set_racing_sim(simulator)
    mpc_planner.set_vehicles_track()

    simulator.sim(sim_time=30.0, two_lap=True, two_lap_name="ego") 
    a_max = ego.system_param.a_max
    v_max = ego.system_param.v_max
    c_max = ego.system_param.curv_max
    print(a_max)
    print(v_max)
    print(c_max)
    simulator.plot_simulation()
    file_name = 'media/animation/mpc_single/a' + str(a_max) + 'v' + str(v_max) + 'c' + str(c_max) + '_' + str(i)+ '.png' 
    plt.savefig(file_name)

def ibr_single(system_param,i):
    track_layout = "oval_shape"  
    track_spec = np.genfromtxt("data/track_layout/" + "oval_shape" + ".csv", delimiter=",") 
    track = racing_env.ClosedTrack(track_spec, track_width=1.0)
    track_center = np.genfromtxt("data/track/" + track_layout + ".csv", delimiter=" ")

    timestep = 1.0 /8.0   #Euler discretization time 
    noise_x = np.maximum(0.3, np.minimum(np.random.randn()+1.3, 2.3)) # randomly starting position
    noise_y = np.maximum(-0.5, np.minimum(np.random.randn()*0.5, 0.5)) 
    
    ego = set_up_ego(timestep, track, track_center,noise_x,noise_y,system_param)
    ibr_planner = set_up_ibr(timestep, track, track_center, ego.system_param)
    ego.set_ctrl_policy(ibr_planner)
 
    # define a simulator
    simulator = offboard.CarRacingSim()
    simulator.set_timestep(timestep)
    simulator.set_track(track,track_center)
    simulator.add_vehicle(ego) #ibr

    ibr_planner.set_racing_sim(simulator)
    ibr_planner.set_vehicles_track()

    simulator.sim(sim_time=30.0, two_lap=True, two_lap_name="ego") 
    a_max = ego.system_param.a_max
    v_max = ego.system_param.v_max
    c_max = ego.system_param.curv_max
    print(a_max)
    print(v_max)
    print(c_max)
    simulator.plot_simulation()
    file_name = 'media/animation/ibr_single/a' + str(a_max) + 'v' + str(v_max) + 'c' + str(c_max) + '_' + str(i)+ '.png' 
    plt.savefig(file_name)

def set_up_ego(timestep, track, track_center,noise_x,noise_y,system_param):
    ego = offboard.DynamicBicycleModel(
        name="ego", param=base.CarParam(edgecolor="black"), system_param=system_param
    )  
    s, ey, epsi, _ = track.get_local_position(noise_x,noise_y,0)
    ego.set_timestep(timestep)                   
    ego.set_state_curvilinear(np.array([s,ey,epsi,0.8]))
    ego.set_state_global(np.array([noise_x,noise_y,0,0.8])) 
    ego.start_logging()
    ego.set_track(track, track_center)
    return ego


def set_up_se_ibr(timestep, track, track_center, system_param):
    se_ibr_param = base.SE_IBR_RacingParam(timestep=timestep)
    racing_game_param = base.RacingGameParam(timestep=timestep, num_horizon_planner=12)
    se_ibr_planner = base.SE_IBR_RacingGame(
        se_ibr_param, racing_game_param = racing_game_param, system_param=system_param
    )
    se_ibr_planner.set_track(track, track_center)                               
    se_ibr_planner.set_timestep(timestep)                          
    return se_ibr_planner

def set_up_ibr(timestep, track, track_center, system_param):
    ibr_param = base.IBR_RacingParam(timestep=timestep)
    racing_game_param = base.RacingGameParam(timestep=timestep, num_horizon_planner=14)
    ibr_planner = base.IBR_RacingGame(
        ibr_param, racing_game_param = racing_game_param, system_param=system_param
    )
    ibr_planner.set_track(track, track_center)                                
    ibr_planner.set_timestep(timestep)                         
    return ibr_planner

def set_up_mpc(timestep, track, track_center, system_param):
    mpc_param = base.MPCRacingParam(timestep=timestep)
    racing_game_param = base.RacingGameParam(timestep=timestep, num_horizon_planner=14)
    mpc_planner = base.MPCRacingGame(
        mpc_param, racing_game_param = racing_game_param, system_param=system_param
    )
    mpc_planner.set_track(track, track_center)                  
    mpc_planner.set_timestep(timestep)                          
    return mpc_planner

class Ego_SystemParam:
    def __init__(self,  a_max=5.8, v_max=3.0, curv_max = 4.0, d_min = 0.8):
        self.a_max = a_max
        self.v_max = v_max
        self.curv_max = curv_max
        self.d_min = d_min

if __name__ == "__main__": 

    system_param = Ego_SystemParam(5.6, 2.2, 4.0, 1.0)
    se_ibr_single(system_param,0)
    ibr_single(system_param,0)
    mpc_single(system_param,0)