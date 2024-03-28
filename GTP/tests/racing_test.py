import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
sys.path.append('C:\\Users\\xshys\\Desktop\\Game_Theoretic_Planner\\GTP')
from utils import base, offboard, racing_env
from utils.constants import *



def se_ibr_overtaking(i):
    track_layout = "oval_shape"   
    track_spec = np.genfromtxt("data/track_layout/" + "oval_shape" + ".csv", delimiter=",") 
    track = racing_env.ClosedTrack(track_spec, track_width=1.0)
    track_center = np.genfromtxt("data/track/" + track_layout + ".csv", delimiter=" ")

    timestep = 1.0 /8.0                                                      # Euler discretization time 
    noise_x_nach = np.maximum(0.1, np.minimum(np.random.randn()+0.3, 0.5))   # randomly starting position
    noise_y_nach = np.maximum(-0.5, np.minimum(np.random.randn(), 0.5)) 
    noise_x_mitte = np.maximum(0.8, np.minimum(np.random.randn()+1.0, 1.2))   
    noise_y_mitte = np.maximum(-0.5, np.minimum(np.random.randn(), 0.5))
    noise_x_vor = np.maximum(1.4, np.minimum(np.random.randn()+1.6,1.8))  
    noise_y_vor = np.maximum(-0.5, np.minimum(np.random.randn(), 0.5)) 
    
    # se_ibr_planner
    ego_system_param = base.Nach_SystemParam()            
    ego = set_up_ego(timestep, track, track_center,noise_x_nach,noise_y_nach,ego_system_param) 
    se_ibr_planner = set_up_se_ibr(timestep, track, track_center, ego.system_param)
    ego.set_ctrl_policy(se_ibr_planner)

    # mpc_planner
    car1_system_param = base.Mitte_SystemParam() 
    car1 = set_up_car("mpc",timestep, track, track_center,noise_x_mitte,noise_y_mitte,car1_system_param)
    mpc_planner = set_up_mpc(timestep, track, track_center, car1.system_param)
    car1.set_ctrl_policy(mpc_planner)

    # ibr_planner
    car2_system_param = base.Vor_SystemParam() 
    car2 = set_up_car("ibr",timestep, track, track_center,noise_x_vor,noise_y_vor,car2_system_param)    
    ibr_planner = set_up_ibr(timestep, track, track_center, car2.system_param)
    car2.set_ctrl_policy(ibr_planner)

    # define a simulator
    simulator = offboard.CarRacingSim()
    simulator.set_timestep(timestep)
    simulator.set_track(track,track_center)
    simulator.add_vehicle(ego)                 #se_ibr
    simulator.add_vehicle(car1)                #mpc
    simulator.add_vehicle(car2)                #ibr

    se_ibr_planner.set_racing_sim(simulator)
    se_ibr_planner.set_vehicles_track()
    mpc_planner.set_racing_sim(simulator)
    mpc_planner.set_vehicles_track()
    ibr_planner.set_racing_sim(simulator)
    ibr_planner.set_vehicles_track()

    simulator.sim(sim_time=30.0, two_lap=True, two_lap_name="ego")
    
    with open("data/racing/overtaking_se_ibr_xcurvs.csv",'a', newline = '')as f:
        writer = csv.writer(f)
        writer.writerow(ego.lap_xcurvs[-1])
    with open("data/racing/overtaking_se_ibr_xglobs.csv",'a', newline = '')as f:
        writer = csv.writer(f)
        writer.writerow(ego.lap_xglobs[-1])
    with open("data/racing/overtaking_se_ibr_laps.csv",'a', newline = '')as f:
        writer = csv.writer(f)
        writer.writerow([ego.laps])  
    with open("data/racing/overtaking_mpc_xcurvs.csv",'a', newline = '')as f:
        writer = csv.writer(f)
        writer.writerow(car1.lap_xcurvs[-1])
    with open("data/racing/overtaking_mpc_xglobs.csv",'a', newline = '')as f:
        writer = csv.writer(f)
        writer.writerow(car1.lap_xglobs[-1])
    with open("data/racing/overtaking_mpc_laps.csv",'a', newline = '')as f:
        writer = csv.writer(f)
        writer.writerow([car1.laps])  
    with open("data/racing/overtaking_ibr_xcurvs.csv",'a', newline = '')as f:
        writer = csv.writer(f)
        writer.writerow(car2.lap_xcurvs[-1])
    with open("data/racing/overtaking_ibr_xglobs.csv",'a', newline = '')as f:
        writer = csv.writer(f)
        writer.writerow(car2.lap_xglobs[-1])
    with open("data/racing/overtaking_ibr_laps.csv",'a', newline = '')as f:
        writer = csv.writer(f)
        writer.writerow([car2.laps]) 

    ea_max = ego.system_param.a_max
    ev_max = ego.system_param.v_max
    ec_max = ego.system_param.curv_max
    print(ea_max)
    print(ev_max)
    print(ec_max)
    o1a_max = car1.system_param.a_max
    o1v_max = car1.system_param.v_max
    o1c_max = car1.system_param.curv_max
    print(o1a_max)
    print(o1v_max)
    print(o1c_max)
    o2a_max = car2.system_param.a_max
    o2v_max = car2.system_param.v_max
    o2c_max = car2.system_param.curv_max
    print(o2a_max)
    print(o2v_max)
    print(o2c_max)
    simulator.plot_simulation()
    file_name1 = 'media/animation/overtaking/se_ibr_a' + str(ea_max) + 'v' + str(ev_max) + 'c' + str(ec_max) \
                                     + '_with_mpc_a'+ str(o1a_max) + 'v' + str(o1v_max) + 'c' + str(o1c_max) \
                                     + '_with_ibr_a'+ str(o2a_max) + 'v' + str(o2v_max) + 'c' + str(o2c_max) + '_'+ str(i)+ '.png'  
    #file_name1 = 'media/animation/overtaking/se_ibr_a' + str(ea_max) + 'v' + str(ev_max) + 'c' + str(ec_max) \
    #                                 + '_with_mpc_a'+ str(o1a_max) + 'v' + str(o1v_max) + 'c' + str(o1c_max) + '_'+ str(i+30)+ '.png'  

    plt.savefig(file_name1)
    
    simulator.plot_state(["ego","mpc","ibr"]) 
    file_name2 = 'media/animation/overtaking/se_ibr_a' + str(ea_max) + 'v' + str(ev_max) + 'c' + str(ec_max) \
                                     + '_with_mpc_a'+ str(o1a_max) + 'v' + str(o1v_max) + 'c' + str(o1c_max) \
                                     + '_with_ibr_a'+ str(o2a_max) + 'v' + str(o2v_max) + 'c' + str(o2c_max) + '_'+ str(i)+ '_states.png' 
    #file_name2 = 'media/animation/overtaking/se_ibr_a' + str(ea_max) + 'v' + str(ev_max) + 'c' + str(ec_max) \
    #                                 + '_with_mpc_a'+ str(o1a_max) + 'v' + str(o1v_max) + 'c' + str(o1c_max) + '_'+ str(i+30)+ '_states.png' 
                                      
    plt.savefig(file_name2)

   

    file_name3 = 'overtaking_se_ibr_a' + str(ea_max) + 'v' + str(ev_max) + 'c' + str(ec_max) \
                       + '_with_mpc_a'+ str(o1a_max) + 'v' + str(o1v_max) + 'c' + str(o1c_max)\
                       + '_with_ibr_a'+ str(o2a_max) + 'v' + str(o2v_max) + 'c' + str(o2c_max)+ '_'+ str(i)
    #file_name3 = 'overtaking_se_ibr_a' + str(ea_max) + 'v' + str(ev_max) + 'c' + str(ec_max) \
    #                   + '_with_mpc_a'+ str(o1a_max) + 'v' + str(o1v_max) + 'c' + str(o1c_max) + '_'+ str(i+30)
                
    simulator.animate(filename=file_name3, ani_time=250, racing_game=True)

def se_ibr_blocking(i):
    track_layout = "oval_shape"   # "m_shape", "l_shape", "ellipse", "goggle"
    track_spec = np.genfromtxt("data/track_layout/" + "oval_shape" + ".csv", delimiter=",") 
    track = racing_env.ClosedTrack(track_spec, track_width=1.0)
    track_center = np.genfromtxt("data/track/" + track_layout + ".csv", delimiter=" ")

    timestep = 1.0 /8.0                                                      # Euler discretization time 
    noise_x_nach = np.maximum(0.1, np.minimum(np.random.randn()+0.3, 0.5))   # randomly starting position
    noise_y_nach = np.maximum(-0.5, np.minimum(np.random.randn(), 0.5)) 
    noise_x_mitte = np.maximum(0.8, np.minimum(np.random.randn()+1.0, 1.2))   
    noise_y_mitte = np.maximum(-0.5, np.minimum(np.random.randn(), 0.5))
    noise_x_vor = np.maximum(1.4, np.minimum(np.random.randn()+1.6,1.8))  
    noise_y_vor = np.maximum(-0.5, np.minimum(np.random.randn(), 0.5)) 
    
    # se_ibr_planner
    ego_system_param = base.Vor_SystemParam()            
    ego = set_up_ego(timestep, track, track_center,noise_x_vor,noise_y_vor,ego_system_param)  
    se_ibr_planner = set_up_se_ibr(timestep, track, track_center, ego.system_param)
    ego.set_ctrl_policy(se_ibr_planner)

    # mpc_planner
    car1_system_param = base.Mitte_SystemParam() 
    car1 = set_up_car("mpc",timestep, track, track_center,noise_x_mitte,noise_y_mitte,car1_system_param)
    mpc_planner = set_up_mpc(timestep, track, track_center, car1.system_param)
    car1.set_ctrl_policy(mpc_planner)

    # ibr_planner
    car2_system_param = base.Nach_SystemParam() 
    car2 = set_up_car("ibr",timestep, track, track_center,noise_x_nach,noise_y_nach,car2_system_param)    
    ibr_planner = set_up_ibr(timestep, track, track_center, car2.system_param)
    car2.set_ctrl_policy(ibr_planner)

    # define a simulator
    simulator = offboard.CarRacingSim()
    simulator.set_timestep(timestep)
    simulator.set_track(track,track_center)
    simulator.add_vehicle(ego)              #se_ibr
    simulator.add_vehicle(car1)             #mpc
    simulator.add_vehicle(car2)             #ibr

    se_ibr_planner.set_racing_sim(simulator)
    se_ibr_planner.set_vehicles_track()
    mpc_planner.set_racing_sim(simulator)
    mpc_planner.set_vehicles_track()
    ibr_planner.set_racing_sim(simulator)
    ibr_planner.set_vehicles_track()

    simulator.sim(sim_time=30.0, two_lap=True, two_lap_name="ego")

    with open("data/racing/blocking_se_ibr_xcurvs.csv",'a', newline = '')as f:
        writer = csv.writer(f)
        writer.writerow(ego.lap_xcurvs[-1])
    with open("data/racing/blocking_se_ibr_xglobs.csv",'a', newline = '')as f:
        writer = csv.writer(f)
        writer.writerow(ego.lap_xglobs[-1])
    with open("data/racing/blocking_se_ibr_laps.csv",'a', newline = '')as f:
        writer = csv.writer(f)
        writer.writerow([ego.laps]) 
    with open("data/racing/blocking_mpc_xcurvs.csv",'a', newline = '')as f:
        writer = csv.writer(f)
        writer.writerow(car1.lap_xcurvs[-1])
    with open("data/racing/blocking_mpc_xglobs.csv",'a', newline = '')as f:
        writer = csv.writer(f)
        writer.writerow(car1.lap_xglobs[-1])
    with open("data/racing/blocking_mpc_laps.csv",'a', newline = '')as f:
        writer = csv.writer(f)
        writer.writerow([car1.laps]) 
    with open("data/racing/blocking_ibr_xcurvs.csv",'a', newline = '')as f:
        writer = csv.writer(f)
        writer.writerow(car2.lap_xcurvs[-1])
    with open("data/racing/blocking_ibr_xglobs.csv",'a', newline = '')as f:
        writer = csv.writer(f)
        writer.writerow(car2.lap_xglobs[-1])
    with open("data/racing/blocking_ibr_laps.csv",'a', newline = '')as f:
        writer = csv.writer(f)
        writer.writerow([car2.laps]) 

    ea_max = ego.system_param.a_max
    ev_max = ego.system_param.v_max
    ec_max = ego.system_param.curv_max
    print(ea_max)
    print(ev_max)
    print(ec_max)
    o1a_max = car1.system_param.a_max
    o1v_max = car1.system_param.v_max
    o1c_max = car1.system_param.curv_max
    print(o1a_max)
    print(o1v_max)
    print(o1c_max)
    o2a_max = car2.system_param.a_max
    o2v_max = car2.system_param.v_max
    o2c_max = car2.system_param.curv_max
    print(o2a_max)
    print(o2v_max)
    print(o2c_max)
    simulator.plot_simulation()
    file_name1 = 'media/animation/blocking/se_ibr_a' + str(ea_max) + 'v' + str(ev_max) + 'c' + str(ec_max) \
                                     + '_with_mpc_a'+ str(o1a_max) + 'v' + str(o1v_max) + 'c' + str(o1c_max) \
                                     + '_with_ibr_a'+ str(o2a_max) + 'v' + str(o2v_max) + 'c' + str(o2c_max) + '_'+ str(i)+ '.png'  
    #file_name1 = 'media/animation/blocking/se_ibr_a' + str(ea_max) + 'v' + str(ev_max) + 'c' + str(ec_max) \
    #                                + '_with_mpc_a'+ str(o1a_max) + 'v' + str(o1v_max) + 'c' + str(o1c_max) + '_'+ str(i+30)+ '.png'  
    plt.savefig(file_name1)
    
    simulator.plot_state(["ego","mpc","ibr"])
    file_name2 = 'media/animation/blocking/se_ibr_a' + str(ea_max) + 'v' + str(ev_max) + 'c' + str(ec_max) \
                                     + '_with_mpc_a'+ str(o1a_max) + 'v' + str(o1v_max) + 'c' + str(o1c_max) \
                                     + '_with_ibr_a'+ str(o2a_max) + 'v' + str(o2v_max) + 'c' + str(o2c_max) + '_'+ str(i)+ '_states.png' 
    #file_name2 = 'media/animation/blocking/se_ibr_a' + str(ea_max) + 'v' + str(ev_max) + 'c' + str(ec_max) \
    #                                 + '_with_mpc_a'+ str(o1a_max) + 'v' + str(o1v_max) + 'c' + str(o1c_max) + '_'+ str(i+30)+  '_states.png' 
    plt.savefig(file_name2)

   
    file_name3 = 'blocking_se_ibr_a' + str(ea_max) + 'v' + str(ev_max) + 'c' + str(ec_max) \
                      + '_with_mpc_a'+ str(o1a_max) + 'v' + str(o1v_max) + 'c' + str(o1c_max)\
                      + '_with_ibr_a'+ str(o2a_max) + 'v' + str(o2v_max) + 'c' + str(o2c_max)+ '_'+ str(i)
    #file_name3 = 'blocking_se_ibr_a' + str(ea_max) + 'v' + str(ev_max) + 'c' + str(ec_max) \
    #                   + '_with_mpc_a'+ str(o1a_max) + 'v' + str(o1v_max) + 'c' + str(o1c_max) + '_'+ str(i+30)

    simulator.animate(filename=file_name3, ani_time=250, racing_game=True)

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

def set_up_car(name,timestep, track, track_center,noise_x,noise_y,system_param):
    car = offboard.DynamicBicycleModel(
        name=name, param=base.CarParam(edgecolor="orange"), system_param=system_param
    )  
    s, ey, epsi, _ = track.get_local_position(noise_x,noise_y,0)
    car.set_timestep(timestep)                    
    car.set_state_curvilinear(np.array([s,ey,epsi,0.8]))
    car.set_state_global(np.array([noise_x,noise_y,0,0.8])) 
    car.start_logging()
    car.set_track(track, track_center)
    return car

def set_up_se_ibr(timestep, track, track_center, system_param):
    se_ibr_param = base.SE_IBR_RacingParam(timestep=timestep)
    racing_game_param = base.RacingGameParam(timestep=timestep, num_horizon_planner=12)
    se_ibr_planner = base.SE_IBR_RacingGame(
        se_ibr_param, racing_game_param = racing_game_param, system_param=system_param
    )
    se_ibr_planner.set_track(track, track_center)                               
    se_ibr_planner.set_timestep(timestep)                         
    return se_ibr_planner

def set_up_mpc(timestep, track, track_center, system_param):
    #time_mpc = 1000 * timestep  #此处 timestep=1.0/10.0
    mpc_param = base.MPCRacingParam(timestep=timestep)
    racing_game_param = base.RacingGameParam(timestep=timestep, num_horizon_planner=12)
    mpc_planner = base.MPCRacingGame(
        mpc_param, racing_game_param = racing_game_param, system_param=system_param
    )
    mpc_planner.set_track(track, track_center)                                
    mpc_planner.set_timestep(timestep)                          
    return mpc_planner

def set_up_ibr(timestep, track, track_center, system_param):
    ibr_param = base.IBR_RacingParam(timestep=timestep)
    racing_game_param = base.RacingGameParam(timestep=timestep, num_horizon_planner=12)
    ibr_planner = base.IBR_RacingGame(
        ibr_param, racing_game_param = racing_game_param, system_param=system_param
    )
    ibr_planner.set_track(track, track_center)                              
    ibr_planner.set_timestep(timestep)                        
    return ibr_planner

if __name__ == "__main__":
    for i in range(1):
        se_ibr_overtaking(i)  
        se_ibr_blocking(i)
        