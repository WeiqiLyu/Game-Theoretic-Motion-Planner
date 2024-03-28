import datetime
import numpy as np
import casadi as ca
from casadi import *
from planner.solver_helper import *
import sys
sys.path.append('C:\\Users\\xshys\\Desktop\\Game_Theoretic_Planner\\GTP')
from utils.constants import *

class MPC_Planner:  
    def __init__(self, racing_game_param):
        self.racing_game_param = racing_game_param
        self.vehicles = None
        self.agent_name = None
        self.track = None
        self.track_center = None

    def get_overtake_flag(self):
        overtake_flag = False
        vehicles_interest = {}
        for name in list(self.vehicles):
            if name != self.agent_name:
                if check_ego_agent_distance(
                    self.vehicles[self.agent_name],                 # ego
                    self.vehicles[name],                            # agent
                    self.racing_game_param,
                    self.track.lap_length,
                ):
                    overtake_flag = True
                    vehicles_interest[name] = self.vehicles[name]
        return overtake_flag, vehicles_interest

    def mpc(
        self,
        xglob_ego,
        time,
        vehicles_interest,
        track_center,
        ):
        start_timer = datetime.datetime.now()
        num_horizon_planner = self.racing_game_param.num_horizon_planner     #Np=10
        timestep  = self.racing_game_param.timestep
        vehicles = self.vehicles
        track = self.track
        track_center = self.track_center   
        ego = self.vehicles[self.agent_name]        
        
        # the maximum and minimum ey's value of the obstacle's predicted trajectory
        num_veh = len(vehicles_interest)
        num = 0
        sorted_vehicles = []
        obs_infos_xcurv = {}
        obs_infos_xglob = {}
        
        # in this list, the vehicle with biggest y will be the first
        for name in list(vehicles_interest):    
            if num == 0:
                sorted_vehicles.append(name)
            elif vehicles_interest[name].xglob[1] >= vehicles_interest[sorted_vehicles[0]].xglob[1]:
                sorted_vehicles.insert(0, name) 
            elif vehicles_interest[name].xglob[1] <= vehicles_interest[sorted_vehicles[0]].xglob[1]:
                sorted_vehicles.append(name) 
            num += 1                   

        for index in range(num_veh):
            name = sorted_vehicles[index]
            if self.vehicles[name].no_dynamics:
                obs_traj_xcurv, obs_traj_xglob = vehicles[name].get_trajectory_nsteps(  #obs_traj为xcurv_est_nsteps
                    time,                              #t0
                    self.racing_game_param.timestep,   #delta_t
                    num_horizon_planner + 1,           #n  
                ) 
            else:
                obs_traj_xcurv, obs_traj_xglob = vehicles[name].get_trajectory_nsteps(num_horizon_planner + 1)        
            obs_infos_xcurv[name] = obs_traj_xcurv
            obs_infos_xglob[name] = obs_traj_xglob               

        self.sorted_vehicles = sorted_vehicles
        self.obs_infos_xcurv = obs_infos_xcurv
        self.obs_infos_xglob = obs_infos_xglob
        self.xglob_ego = xglob_ego                
        solution_x, solution_xpoly, solution_ypoly = self.solve_optimization_problem(
            xglob_ego,
            sorted_vehicles,
            obs_infos_xcurv,
            obs_infos_xglob,
            num_horizon_planner,
            num_veh,
            ego,
            track,
            track_center,
            timestep)
        end_timer = datetime.datetime.now()
        solver_time = (end_timer - start_timer).total_seconds()
        print("mpc planner solver time: {}".format(solver_time))

        num_sampling_per_section = 20
        poly_traj_points = np.zeros((num_horizon_planner,num_sampling_per_section, 2))
        for j in range(num_horizon_planner):             
            for i in range(num_sampling_per_section): 
                t = j * timestep  + i * timestep/num_sampling_per_section
                poly_traj_points[j,i,:] = get_polynomial_trajectory(solution_xpoly[:,j], solution_ypoly[:,j], t) 
        return (
            solution_x,
            poly_traj_points,
            solution_xpoly,
            solution_ypoly,
            solver_time,
            )

    def solve_optimization_problem(
        self,
        xglob_ego,
        sorted_vehicles,
        obs_infos_xcurv,
        obs_infos_xglob,
        num_horizon_planner,
        num_veh,
        ego,
        track,
        track_center,
        timestep
    ):
        selected_idx = np.zeros((num_horizon_planner+1,1))
        X_track = []
        Y_track = []
        psi_track = []
        curv_track = []
        ey =[]
        tangent_vector = np.zeros((num_horizon_planner+1,2))
        normal_vector = np.zeros((num_horizon_planner+1,2))
  
        # initialize the problem      
        opti = ca.Opti()    
        
        # define variables
        x = opti.variable(X_DIM, num_horizon_planner + 1)  #[x,y,psi,v]  
        xpoly = opti.variable(3, num_horizon_planner)      #[a0,a1,a2]
        ypoly = opti.variable(3, num_horizon_planner)      #[b0,b1,b2]                          

        # add constraints and cost function
        opti.subject_to(x[:,0] == xglob_ego)
        cost_mpc = 0    
        
        for i in range(num_horizon_planner):
            t0 = timestep * i
            tf = timestep * (i+1)                    
            
            # continuity constraints
            # waypoint position constraint
            opti.subject_to (
                [
                xpoly[0,i]+ xpoly[1,i] * t0 + xpoly[2,i] * t0 **2 - x[0,i] == 0,         # ai_0 + ai_1*t0 + ai_2*t0^2 = xi_t0
                xpoly[0,i]+ xpoly[1,i] * tf + xpoly[2,i] * tf **2 - x[0,i+1]== 0,        # ai_0 + ai_1*tf + ai_2*tf^2 = xi_tf
                ypoly[0,i]+ ypoly[1,i] * t0 + ypoly[2,i] * t0 **2 - x[1,i]== 0,          # bi_0 + bi_1*t0 + bi_2*t0^2 = yi_t0
                ypoly[0,i]+ ypoly[1,i] * tf + ypoly[2,i] * tf **2 - x[1,i+1]== 0,]       # bi_0 + bi_1*tf + bi_2*tf^2 = yi_tf)
                ) 

            # waypoint velocity constraint
            xdot_i = x[3,i]*cos(x[2,i])         #conversion in global koor.
            xdot_i1 = x[3,i+1]*cos(x[2,i+1])
            ydot_i = x[3,i]*sin(x[2,i]) 
            ydot_i1 = x[3,i+1]*sin(x[2,i+1])

            opti.subject_to (                               
                [xpoly[1,i] + 2.0*xpoly[2,i]*t0 - xdot_i == 0,   # ai_1 + 2*ai_2*t0 = vxi_t0
                xpoly[1,i] + 2.0*xpoly[2,i]*tf - xdot_i1 == 0,   # ai_1 + 2*ai_2*tf = vxi_tf
                ypoly[1,i] + 2.0*ypoly[2,i]*t0 - ydot_i == 0,    # bi_1 + 2*bi_2*t0 = vyi_t0
                ypoly[1,i] + 2.0*ypoly[2,i]*tf - ydot_i1 == 0]   # bi_1 + 2*bi_2*tf = vyi_tf
                )

            # acceleration constraint
            opti.subject_to((2.0*xpoly[2,i])**2 + (2.0*ypoly[2,i])**2 <= ego.system_param.a_max**2)
        
        for i in range(int(num_horizon_planner/2)):
            # curvature constraint
            curv = 2*(xpoly[1,i]*ypoly[2,i]-xpoly[2,i]*ypoly[1,i])
            opti.subject_to( curv <= ego.system_param.curv_max *(x[3,i]**3) )
            opti.subject_to( -ego.system_param.curv_max *(x[3,i]**3)<=curv)
                             
        for i in range(num_horizon_planner + 1):    
            # speed constraint
            opti.subject_to(x[3,i]<= ego.system_param.v_max)

            # track constraint                
            # player i's current guess of its optimal strategy
            smallest_idx = find_smallest_index(xglob_ego,track_center) 
            selected_idx[i] = mod(np.floor(smallest_idx + ego.system_param.v_max*timestep*100*i),track_center.shape[0])
            X_track_i, Y_track_i, psi_track_i, curv_track_i, tangent_vector[i], normal_vector[i] = \
                find_closest_point(int(selected_idx[i]),track_center)
            X_track.append(X_track_i)
            Y_track.append(Y_track_i)
            psi_track.append(psi_track_i)
            curv_track.append(curv_track_i)
            ey_i = normal_vector[i,0]*(x[0,i]- X_track[i])+ normal_vector[i,1]*(x[1,i]-Y_track[i])
            ey.append(ey_i) 
            opti.subject_to(ey[i] <= track.width-ego.param.width)
            opti.subject_to(ey[i] >= -track.width+ego.param.width)
 
            # collision avoidance constraint
            for pos_index in range(num_veh):
                name = sorted_vehicles[pos_index ]                
                obs_traj_xglob = obs_infos_xglob[name]            #dim(obs_traj_xglob)=4*(num_horizon_planner+1)
                obs_traj_xcurv = obs_infos_xcurv[name]
                while obs_traj_xcurv[0, i] > track.lap_length:
                    obs_traj_xcurv[0, i] = obs_traj_xcurv[0, i] - track.lap_length
                diffx = x[0, i] - obs_traj_xglob[0, i]
                diffy = x[1, i] - obs_traj_xglob[1, i]  
                opti.subject_to(diffx**2 + diffy**2 >= ego.system_param.d_min**2)
  
        # linear aprroximation of mpc_cost
        s_lin_N = tangent_vector[num_horizon_planner,0]*x[0,num_horizon_planner] +\
                  tangent_vector[num_horizon_planner,1]*x[1,num_horizon_planner]

        cost_mpc -= 1*s_lin_N 

        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}   
        
        solution_x = np.zeros((X_DIM, num_horizon_planner + 1))
        solution_xpoly = np.zeros((3, num_horizon_planner ))
        solution_ypoly = np.zeros((3, num_horizon_planner ))
        
        for j in range(num_horizon_planner+1):  
            x_j = j * ego.xglob[3]*cos(ego.xglob[2]) * timestep + ego.xglob[0]
            y_j = j * ego.xglob[3]*sin(ego.xglob[2]) * timestep + ego.xglob[1]
            psi_j = psi_track[j]
            
            # set initial value of v
            opti.set_initial(x[3, j], ego.xglob[3])
            # set initial value of psi
            opti.set_initial(x[2, j], psi_j)              
            # set initial value of x
            opti.set_initial(x[0, j], x_j)
            # set initial value of y
            opti.set_initial(x[1, j], y_j)
 
        
        for j in range(num_horizon_planner):  
            opti.set_initial(xpoly[0,j],0.01)
            opti.set_initial(xpoly[1,j],0.01)
            opti.set_initial(xpoly[2,j],0.01)
            opti.set_initial(ypoly[0,j],0.02)
            opti.set_initial(ypoly[1,j],0.02)
            opti.set_initial(ypoly[2,j],0.02)

        opti.minimize(cost_mpc)
    
        opti.solver("ipopt", option)
        try:
            sol = opti.solve()
            solution_x = sol.value(x)
            solution_xpoly = sol.value(xpoly)
            solution_ypoly = sol.value(ypoly)
            cost_mpc = sol.value(cost_mpc)
        except RuntimeError:
            solution_x = opti.debug.value(x)
            solution_xpoly = opti.debug.value(xpoly)
            solution_ypoly = opti.debug.value(ypoly)
            cost_mpc = float("inf")
            print("solver fail to find the solution, the non-converged solution is used")

        return solution_x, solution_xpoly, solution_ypoly