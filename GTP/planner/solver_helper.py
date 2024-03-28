import numpy as np
import copy
import sys
sys.path.append('C:\\Users\\xshys\\Desktop\\Game_Theoretic_Planner\\GTP')
from utils.constants import *

def check_ego_agent_distance(ego, agent, racing_game_param, lap_length):
    vehicle_interest = False
    delta_s = abs(ego.xcurv[0] - agent.xcurv[0])   # delta_s = abs(s_ego - s_agent) 
    s_agent = copy.deepcopy(agent.xcurv[0])
    s_ego = copy.deepcopy(ego.xcurv[0])
    while s_agent > lap_length:                         
        s_agent = s_agent - lap_length
    while s_ego > lap_length:
        s_ego = s_ego - lap_length
    if (
        # agent and ego in same lap, agent is in front of the ego
        (
            (
                s_agent - s_ego
                <= racing_game_param.safety_factor * ego.param.length    
                + racing_game_param.planning_prediction_factor * delta_s
            )
            and (s_agent >= s_ego)
        )
        or (
            # agent is in next lap, agent is in front of the ego
            (
                s_agent + lap_length - s_ego
                <= racing_game_param.safety_factor * ego.param.length
                + racing_game_param.planning_prediction_factor * delta_s
            )
            and (s_agent + lap_length >= s_ego)
        )
        or (
            # agent and ego in same lap, ego is in front of the agent
            (
                -s_agent + s_ego
                <= 1.0 * ego.param.length               
                + 0 * racing_game_param.planning_prediction_factor * delta_s
            )
            and (s_agent <= s_ego)
        )
        or (
            # ego is in next lap, ego is in front of the agent
            (
                -s_agent + s_ego + lap_length
                <= 1.0 * ego.param.length
                + 0 * racing_game_param.planning_prediction_factor * delta_s
            )
            and (s_agent <= s_ego + lap_length)
        )
    ):
        vehicle_interest = True  # a surrounding vehicle is in the ego vehicle's range of overtaking
    return vehicle_interest

def find_smallest_index(current_position, track_center):
    posX = current_position[0]
    posY = current_position[1]
    smallest_distance = 9999
    smallest_idx = 0
    num_points = track_center.shape[0]                          
    for i in range(num_points):
        search_idx = i     
        search_distance = (track_center[search_idx][0] - posX)**2 +\
            (track_center[search_idx][1] - posY)**2
        if search_distance < smallest_distance:
            smallest_distance = search_distance
            smallest_idx = search_idx
    return smallest_idx

def find_closest_point(smallest_idx, track_center):
    closest_point = track_center[smallest_idx]
    X_track = closest_point[0]
    Y_track = closest_point[1]
    psi_track = closest_point[2]
    curv_track = closest_point[3]
    tangent_vector = np.array([np.cos(psi_track),np.sin(psi_track)])
    normal_vector = np.array([-np.sin(psi_track),np.cos(psi_track)])
    return X_track, Y_track, psi_track, curv_track, tangent_vector, normal_vector
 
def get_polynomial_trajectory(xpoly, ypoly, t): 
    a0, a1, a2 = xpoly[:]
    b0, b1, b2 = ypoly[:]
    polynomial_x = (a0 + a1 * t + a2 * (t ** 2))  
    polynomixl_y = (b0 + b1 * t + b2 * (t ** 2))
    return [polynomial_x, polynomixl_y]

def get_ref_state_input(ego,fout):
    """
    given a flat output trajectory[x,y], generate reference states and inputs using differential flatness
    """       
    sig = fout[0,:]                # sig = [x_flat,y_flat].T
    sigd1 = fout[1,:]              # sigd1 =  erste derivative of sig 
    sigd2 = fout[2,:]              # sigd2 =  second derivative of sig
        
    # params that needs to be feed into differential flatness computation
    L = ego.param.dynamics_param.L                # the distance between front and rear axle of the vehicle
        
    # reference states [x,y,theta,v] 
    x = sig[0]
    y = sig[1]
    psi = np.arctan2(sigd1[1],sigd1[0])
    v = np.linalg.norm([sigd1[0],sigd1[1]])

    # reference inputs [a,phi]
    a = (sigd1[0]*sigd2[0] + sigd1[1]*sigd2[1])/np.linalg.norm([sigd1[0], sigd1[1]])
    curv = (sigd1[0]*sigd2[1] - sigd1[1]*sigd2[0])/np.linalg.norm([sigd1[0], sigd1[1]])**3
    delta = np.arctan2(L*curv,1)

    return x, y, psi, v, a, delta
