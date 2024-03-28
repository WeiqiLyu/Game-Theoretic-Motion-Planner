import numpy as np
import sympy as sp
import sys
sys.path.append('C:\\Users\\xshys\\Desktop\\Game_Theoretic_Planner\\GTP')
from utils.constants import *
from planner import se_ibr_planner, ibr_planner, mpc_planner, solver_helper
import copy

# Base Controller
class ControlBase:
    def __init__(self):
        self.agent_name = None             
        self.time = 0.0                    
        self.timestep = None              
        self.xcurv = None                      
        self.xglob = None                 
        self.u = None
        self.realtime_flag = False
        # store the information (e.g. states, inputs) of current lap
        self.lap_times, self.lap_xcurvs, self.lap_xglobs, self.lap_inputs = [], [], [], []
        self.lap_times.append(self.time)   #=[0.0,0.1,0.2,...]
        # store the information (e.g. state, inputs) of the whole simulation
        self.times, self.xglobs, self.xcurvs, self.inputs = [], [], [], []
        self.laps = 0                      
        self.track = None                 
        self.track_center = None           

    def set_track(self, track, track_center):
        self.track = track
        self.lap_length = track.lap_length
        self.point_and_tangent = track.point_and_tangent
        self.lap_width = track.width
        self.track_center = track_center

    def set_racing_sim(self, racing_sim):
        self.racing_sim = racing_sim

    def set_timestep(self, timestep):
        self.timestep = timestep

    def set_state(self, xcurv, xglob):
        self.xcurv = xcurv                       
        self.xglob = xglob                       

    def calc_input(self):
        pass

    def get_input(self):
        return self.u
    
    def update_memory(self, current_lap): 
        xcurv = copy.deepcopy(self.xcurv)
        xglob = copy.deepcopy(self.xglob)
        time = copy.deepcopy(self.time)
        if  xcurv[0] > self.lap_length * (current_lap + 1): 
            self.lap_xglobs.append(xglob)
            self.lap_times.append(time)
            self.lap_xcurvs.append(xcurv)
            self.lap_inputs.append(self.u)

            self.xglobs.append(self.lap_xglobs)
            self.times.append(self.lap_times)
            self.xcurvs.append(self.lap_xcurvs)
            self.inputs.append(self.lap_inputs)
            xcurv = copy.deepcopy(self.xcurv)
            xcurv[0] = xcurv[0] - self.lap_length * (current_lap + 1)  #为什么这样写？
            self.laps = self.laps + 1

            self.lap_xglobs, self.lap_xcurvs, self.lap_inputs, self.lap_times = [], [], [], []
            self.lap_xglobs.append(xglob)
            self.lap_times.append(time)
            self.lap_xcurvs.append(xcurv)
        else:
            xcurv[0] = xcurv[0] - current_lap * self.lap_length
            self.lap_xglobs.append(xglob)
            self.lap_times.append(time)    #lap_times = [0,0.1,0.2,...]
            self.lap_xcurvs.append(xcurv)
            self.lap_inputs.append(self.u)


class RacingGameParam:
    def __init__(
        self,
        safety_factor=4.5,
        num_horizon_ctrl=10,
        num_horizon_planner=12,
        planning_prediction_factor=0.5, 
        alpha=0.3,             # tuable parameter in sensitivity enhanced term
        mue = 0.1,             # Lagrange Multiplier                   
        timestep=None,
    ):
        self.safety_factor = safety_factor
        self.num_horizon_ctrl = num_horizon_ctrl
        self.num_horizon_planner = num_horizon_planner
        self.planning_prediction_factor = planning_prediction_factor
        self.alpha = alpha
        self.timestep = timestep
        self.mue = mue


class MPCRacingParam:
    def __init__(
        self,
        num_horizon_planner=12,            
        timestep=None,                                         
    ):
        self.num_horizon_planner = num_horizon_planner
        self.timestep = timestep

class MPCRacingGame(ControlBase):
    def __init__(self, mpc_param, racing_game_param, system_param=None):
        ControlBase.__init__(self)
        self.mpc_param = mpc_param                     
        self.racing_game_param = racing_game_param     
        self.system_param = system_param              
        self.x_pred = None                             
        self.u_pred = np.zeros((self.mpc_param.num_horizon_planner,2))
        self.mpc_planner = mpc_planner.MPC_Planner(racing_game_param)    
        # Initialize the controller iteration
        self.iter = 0                           
        self.time_in_iter = 0                    
        self.openloop_prediction = None
        self.realtime_flag = False

    def set_vehicles_track(self):
        if self.realtime_flag == False:
            vehicles = self.racing_sim.vehicles       
            self.mpc_planner.track = self.track
            self.mpc_planner.track_center =self.track_center
        else:
            vehicles = self.vehicles
        self.mpc_planner.vehicles = vehicles

    def calc_input(self):
        self.mpc_planner.agent_name = self.agent_name 
        timestep = self.mpc_param.timestep
        ego = self.mpc_planner.vehicles[self.agent_name]
        num_horizon_planner = self.mpc_param.num_horizon_planner
        xcurv = copy.deepcopy(self.xcurv)                    
        xglob = copy.deepcopy(self.xglob)
        dx = np.zeros((num_horizon_planner,1))
        ddx = np.zeros((num_horizon_planner,1))
        dy = np.zeros((num_horizon_planner,1))
        ddy = np.zeros((num_horizon_planner,1))
        flat_output = np.zeros((num_horizon_planner, 3, 2))
        x = np.zeros((num_horizon_planner,1))
        y = np.zeros((num_horizon_planner,1))
        psi = np.zeros((num_horizon_planner,1))
        v = np.zeros((num_horizon_planner,1))
        a = np.zeros((num_horizon_planner,1))
        delta = np.zeros((num_horizon_planner,1))
        while xcurv[0] > self.lap_length:                 
            xcurv[0] = xcurv[0] - self.lap_length
        overtake_flag, vehicles_interest = self.mpc_planner.get_overtake_flag()
 
        (
            solution_x,
            poly_traj_points,
            solution_xpoly,
            solution_ypoly,
            solver_time,
        ) = self.mpc_planner.mpc(xglob,self.time,vehicles_interest,self.track_center)
        
        for i in range(num_horizon_planner):
            dx[i] = solution_xpoly[1,i] + 2 * solution_xpoly[2,i] * timestep * i
            ddx[i] =  2 * solution_xpoly[2,i]
            dy[i] = solution_ypoly[1,i] + 2 * solution_ypoly[2,i] * timestep * i
            ddy[i] =  2 * solution_ypoly[2,i]        
            flat_output[i,0,0] =  solution_x[0,i]
            flat_output[i,0,1] =  solution_x[1,i]
            flat_output[i,1,0] =  dx[i]
            flat_output[i,1,1] =  dy[i]
            flat_output[i,2,0] =  ddx[i]
            flat_output[i,2,1] =  ddy[i]
            x[i], y[i], psi[i], v[i], a[i], delta[i] = solver_helper.get_ref_state_input(ego, flat_output[i])
            self.u_pred[i,0] = a[i]         #longitudinal acceleration
            self.u_pred[i,1] = delta[i]
        
        self.x_pred = solution_x.T
        self.u = self.x_pred[1,:]            # use the predicted states as the states in next step
        #self.xglob_pred = self.x_pred[0,:]  # open-loop control inputs from endogenous transformation
        #self.u = self.u_pred[0, :]
        
        self.mpc_planner.vehicles[self.agent_name].vehicles_interest.append(vehicles_interest)
        self.mpc_planner.vehicles[self.agent_name].se_ibr_prediction.append(None)
        self.mpc_planner.vehicles[self.agent_name].ibr_prediction.append(None) 
        self.mpc_planner.vehicles[self.agent_name].mpc_prediction.append(self.x_pred) 
        self.mpc_planner.vehicles[self.agent_name].solver_time.append(solver_time)

        self.time += self.timestep
    

class IBR_RacingParam:
    def __init__(
        self,
        num_horizon_planner=12,            # horizon of trajectory planner Np =12
        timestep=None,                    
    ):
        self.num_horizon_planner = num_horizon_planner
        self.timestep = timestep

class IBR_RacingGame(ControlBase):
    def __init__(self, ibr_param, racing_game_param, system_param=None):
        ControlBase.__init__(self)
        self.ibr_param = ibr_param                    
        self.racing_game_param = racing_game_param    
        self.system_param = system_param              
        self.x_pred = None                            
        self.u_pred = np.zeros((self.ibr_param.num_horizon_planner,2))
        self.ibr_planner = ibr_planner.IBR_Planner(racing_game_param)   
        self.realtime_flag = False

    def set_vehicles_track(self):
        if self.realtime_flag == False:
            vehicles = self.racing_sim.vehicles        
            self.ibr_planner.track = self.track
            self.ibr_planner.track_center =self.track_center
        else:
            vehicles = self.vehicles
        self.ibr_planner.vehicles = vehicles

    def calc_input(self):
        self.ibr_planner.agent_name = self.agent_name 
        timestep = self.ibr_param.timestep
        ego = self.ibr_planner.vehicles[self.agent_name]
        num_horizon_planner = self.ibr_param.num_horizon_planner
        xcurv = copy.deepcopy(self.xcurv)                    
        xglob = copy.deepcopy(self.xglob)
        dx = np.zeros((num_horizon_planner,1))
        ddx = np.zeros((num_horizon_planner,1))
        dy = np.zeros((num_horizon_planner,1))
        ddy = np.zeros((num_horizon_planner,1))
        flat_output = np.zeros((num_horizon_planner, 3, 2))
        x = np.zeros((num_horizon_planner,1))
        y = np.zeros((num_horizon_planner,1))
        psi = np.zeros((num_horizon_planner,1))
        v = np.zeros((num_horizon_planner,1))
        a = np.zeros((num_horizon_planner,1))
        delta = np.zeros((num_horizon_planner,1))
        while xcurv[0] > self.lap_length:                 
            xcurv[0] = xcurv[0] - self.lap_length

        overtake_flag, vehicles_interest = self.ibr_planner.get_overtake_flag()
 
        (
            target_traj_xglob,
            solution_xpoly,
            solution_ypoly,
            solver_time,          
        ) = self.ibr_planner.ibr( self.time, vehicles_interest)
        
        for i in range(num_horizon_planner):
            dx[i] = solution_xpoly[1,i] + 2 * solution_xpoly[2,i] * timestep * i
            ddx[i] =  2 * solution_xpoly[2,i]
            dy[i] = solution_ypoly[1,i] + 2 * solution_ypoly[2,i] * timestep * i
            ddy[i] =  2 * solution_ypoly[2,i]        
            flat_output[i,0,0] =  target_traj_xglob[0,i]
            flat_output[i,0,1] =  target_traj_xglob[1,i]
            flat_output[i,1,0] =  dx[i]
            flat_output[i,1,1] =  dy[i]
            flat_output[i,2,0] =  ddx[i]
            flat_output[i,2,1] =  ddy[i]
            x[i], y[i], psi[i], v[i], a[i], delta[i] = solver_helper.get_ref_state_input(ego, flat_output[i])
            self.u_pred[i,0] = a[i]   #longitudinal acceleration
            self.u_pred[i,1] = delta[i]
        
        self.x_pred = target_traj_xglob.T
        self.u = self.x_pred[1,:]
        #self.xglob_pred = self.x_pred[0,:]
        #self.u = self.u_pred[0, :]

        self.ibr_planner.vehicles[self.agent_name].vehicles_interest.append(vehicles_interest)
        self.ibr_planner.vehicles[self.agent_name].solver_time.append(solver_time)
        self.ibr_planner.vehicles[self.agent_name].se_ibr_prediction.append(None)
        self.ibr_planner.vehicles[self.agent_name].ibr_prediction.append(self.x_pred) #开环x_glob值
        self.ibr_planner.vehicles[self.agent_name].mpc_prediction.append(None)

        self.time += self.timestep



class SE_IBR_RacingParam:
    def __init__(
        self,
        num_horizon_planner=12,            # horizon of trajectory planner Np =12
        timestep=None,                     

    ):
        self.num_horizon_planner = num_horizon_planner
        self.timestep = timestep

class SE_IBR_RacingGame(ControlBase):
    def __init__(self, se_ibr_param, racing_game_param, system_param=None):
        ControlBase.__init__(self)
        self.se_ibr_param = se_ibr_param                     
        self.racing_game_param = racing_game_param    
        self.system_param = system_param               
        self.x_pred = None                             
        self.u_pred = np.zeros((self.se_ibr_param.num_horizon_planner,2))
        self.se_ibr_planner = se_ibr_planner.SE_IBR_Planner(racing_game_param)   
        self.realtime_flag = False

    def set_vehicles_track(self):
        if self.realtime_flag == False:
            vehicles = self.racing_sim.vehicles        
            self.se_ibr_planner.track = self.track
            self.se_ibr_planner.track_center =self.track_center
        else:
            vehicles = self.vehicles
        self.se_ibr_planner.vehicles = vehicles

    def calc_input(self):
        self.se_ibr_planner.agent_name = self.agent_name
        timestep = self.se_ibr_param.timestep
        ego = self.se_ibr_planner.vehicles[self.agent_name]
        num_horizon_planner = self.se_ibr_param.num_horizon_planner
        alpha = self.racing_game_param.alpha
        mue = self.racing_game_param.mue
        xcurv = copy.deepcopy(self.xcurv)                    
        xglob = copy.deepcopy(self.xglob)
        dx = np.zeros((num_horizon_planner,1))
        ddx = np.zeros((num_horizon_planner,1))
        dy = np.zeros((num_horizon_planner,1))
        ddy = np.zeros((num_horizon_planner,1))
        flat_output = np.zeros((num_horizon_planner, 3, 2))
        x = np.zeros((num_horizon_planner,1))
        y = np.zeros((num_horizon_planner,1))
        psi = np.zeros((num_horizon_planner,1))
        v = np.zeros((num_horizon_planner,1))
        a = np.zeros((num_horizon_planner,1))
        delta = np.zeros((num_horizon_planner,1))
        while xcurv[0] > self.lap_length:                
            xcurv[0] = xcurv[0] - self.lap_length

        overtake_flag, vehicles_interest = self.se_ibr_planner.get_overtake_flag()
 
        (
            solution_x,
            solution_xpoly,
            solution_ypoly,
            solver_time,
        ) = self.se_ibr_planner.se_ibr( self.time, vehicles_interest,alpha,mue)
        
        for i in range(num_horizon_planner):
            dx[i] = solution_xpoly[1,i] + 2 * solution_xpoly[2,i] * timestep * i
            ddx[i] =  2 * solution_xpoly[2,i]
            dy[i] = solution_ypoly[1,i] + 2 * solution_ypoly[2,i] * timestep * i
            ddy[i] =  2 * solution_ypoly[2,i]        
            flat_output[i,0,0] =  solution_x[0,i]
            flat_output[i,0,1] =  solution_x[1,i]
            flat_output[i,1,0] =  dx[i]
            flat_output[i,1,1] =  dy[i]
            flat_output[i,2,0] =  ddx[i]
            flat_output[i,2,1] =  ddy[i]
            x[i], y[i], psi[i], v[i], a[i], delta[i] = solver_helper.get_ref_state_input(ego, flat_output[i])
            self.u_pred[i,0] = a[i]   #longitudinal acceleration
            self.u_pred[i,1] = delta[i]
        
        self.x_pred = solution_x.T
        self.u = self.x_pred[1,:]
        #self.xglob_pred = self.x_pred[0,:]
        #self.u = self.u_pred[0, :]
        
        self.se_ibr_planner.vehicles["ego"].vehicles_interest.append(vehicles_interest)
        self.se_ibr_planner.vehicles["ego"].solver_time.append(solver_time)
        self.se_ibr_planner.vehicles["ego"].se_ibr_prediction.append(self.x_pred) 
        self.se_ibr_planner.vehicles["ego"].ibr_prediction.append(None)
        self.se_ibr_planner.vehicles["ego"].mpc_prediction.append(None)
        self.time += self.timestep

# Base Racing Car
class BicycleDynamicsParam:
    def __init__(
        self,
        m=1.98,
        lf=0.125,
        lr=0.125,
        Iz=0.024,
        Df=0.8 * 1.98 * 9.81 / 2.0,
        Cf=1.25,
        Bf=1.0,
        Dr=0.8 * 1.98 * 9.81 / 2.0,
        Cr=1.25,
        Br=1.0,
        L = 0.25,                  # L= lf +lr
    ):
        self.m = m
        self.lf = lf
        self.lr = lr
        self.Iz = Iz
        self.Df = Df
        self.Cf = Cf
        self.Bf = Bf
        self.Dr = Dr
        self.Cr = Cr
        self.Br = Br
        self.L = L                 # the distance between front and rear axle of the vehicle

    def get_params(self):
        return (
            self.m,
            self.lf,
            self.lr,
            self.Iz,
            self.Df,
            self.Cf,
            self.Bf,
            self.Dr,
            self.Cr,
            self.Br,
            self.L,
        )


class CarParam:
    def __init__(self, length=0.4, width=0.2, facecolor="None", edgecolor="black"):
        self.length = length
        self.width = width
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.dynamics_param = BicycleDynamicsParam()

class Vor_SystemParam:
    def __init__(self, delta_max=0.5, a_max=5.6, v_max=1.8, v_min=0, curv_max = 4, d_min = 1.0):
        self.delta_max = delta_max
        self.a_max = a_max
        self.v_max = v_max
        self.v_min = v_min
        self.curv_max = curv_max
        self.d_min = d_min

class Mitte_SystemParam:
    def __init__(self, delta_max=0.5, a_max=5.6, v_max=2.0, v_min=0, curv_max = 4, d_min = 1.0):
        self.delta_max = delta_max
        self.a_max = a_max
        self.v_max = v_max
        self.v_min = v_min
        self.curv_max = curv_max
        self.d_min = d_min

class Nach_SystemParam:
    def __init__(self, delta_max=0.5, a_max=5.6, v_max=2.2, v_min=0, curv_max = 4, d_min = 1.0):
        self.delta_max = delta_max
        self.a_max = a_max
        self.v_max = v_max
        self.v_min = v_min
        self.curv_max = curv_max
        self.d_min = d_min


class ModelBase:
    def __init__(self, name=None, param=None, no_dynamics=False, system_param=None):
        self.name = name                 
        self.param = param              
        self.system_param = system_param 
        self.no_dynamics = False
        self.time = 0.0                  
        self.timestep = None             
        self.xcurv = None               
        self.xglob = None                
        self.u = None
        self.zero_noise_flag = False     
        self.lap_times = []
        if self.no_dynamics:
            pass 
        else:     
            self.lap_times.append(self.time)  
        self.lap_xcurvs, self.lap_xglobs, self.lap_inputs = [], [], []
        self.times, self.xglobs, self.xcurvs, self.inputs = [], [], [], []
        self.laps = 0
        self.realtime_flag = False
        self.xglob_log = []
        self.xcurv_log = []
        self.vehicles_interest = []
        self.solver_time = []
        self.mpc_prediction = []
        self.se_ibr_prediction = []
        self.ibr_prediction = []

    def set_zero_noise(self):
        self.zero_noise_flag = True

    def set_timestep(self, dt):
        self.timestep = dt

    def set_state_curvilinear(self, xcurv):
        self.xcurv = xcurv

    def set_state_global(self, xglob):
        self.xglob = xglob

    def start_logging(self):
        self.lap_xcurvs, self.lap_xglobs, self.lap_inputs = [], [], []
        self.lap_xcurvs.append(copy.deepcopy(self.xcurv))
        self.lap_xglobs.append(copy.deepcopy(self.xglob))

    def set_track(self, track, track_center):
        self.track = track                     
        self.lap_length = track.lap_length
        self.point_and_tangent = track.point_and_tangent
        self.lap_width = track.width
        self.track_center = track_center

    def set_ctrl_policy(self, ctrl_policy):
        self.ctrl_policy = ctrl_policy          
        self.ctrl_policy.agent_name = self.name  

    def calc_ctrl_input(self):
        self.ctrl_policy.set_state(self.xcurv, self.xglob)
        self.ctrl_policy.calc_input()
        self.u = self.ctrl_policy.get_input()

    def forward_dynamics(self):
        pass

    def forward_one_step(self, realtime_flag):
        if self.no_dynamics:
            self.forward_dynamics()
            self.update_memory()
        elif realtime_flag == False:
            self.calc_ctrl_input()
            self.forward_dynamics(realtime_flag)
            self.update_memory()
        elif realtime_flag == True:
            self.forward_dynamics(realtime_flag)

    def update_memory(self):
        xcurv = copy.deepcopy(self.xcurv)  
        xglob = copy.deepcopy(self.xglob)
        time = copy.deepcopy(self.time)
        self.xglob_log.append(xglob)
        self.xcurv_log.append(xcurv)
        if  np.abs(xcurv[0]-self.lap_xcurvs[-1][0]) > 10 : 
            self.lap_xglobs.append(xglob)
            self.lap_times.append(time)
            self.lap_xcurvs.append(xcurv)      
            self.lap_inputs.append(self.u)
            self.xglobs.append(self.lap_xglobs) 
            self.times.append(self.lap_times)
            self.xcurvs.append(self.lap_xcurvs)
            self.inputs.append(self.lap_inputs)
            #self.xcurv[0] = self.xcurv[0] - self.lap_length  
            self.laps = self.laps + 1           
            self.lap_xglobs, self.lap_xcurvs, self.lap_inputs, self.lap_times = [], [], [], []
            self.lap_xglobs.append(xglob)  
            self.lap_times.append(time)
            self.lap_xcurvs.append(xcurv)  
        else:
            self.lap_xglobs.append(xglob)
            self.lap_times.append(time)
            self.lap_xcurvs.append(xcurv)
            self.lap_inputs.append(self.u)
        print(self.laps)

class NoDynamicsModel(ModelBase):
    def __init__(self, name=None, param=None, xcurv=None, xglob=None):
        ModelBase.__init__(self, name=name, param=param)
        self.no_dynamics = True

    def set_state_curvilinear_func(self, t_symbol, s_func, ey_func):
        self.t_symbol = t_symbol
        self.s_func = s_func
        self.ey_func = ey_func
        self.xcurv = np.zeros((X_DIM,))
        self.xglob = np.zeros((X_DIM,))
        self.xcurv, self.xglob = self.get_estimation(0)

    def get_estimation(self, t0):
        # position estimation in curvilinear coordinates  
        xcurv_est = np.zeros((X_DIM,)) 
        xcurv_est[3] = sp.diff(self.s_func, self.t_symbol).subs(self.t_symbol, t0)    
        xcurv_est[2] = 0
        xcurv_est[1] = self.ey_func.subs(self.t_symbol, t0) 
        xcurv_est[0] = self.s_func.subs(self.t_symbol, t0) 

        # position estimation in global coordinates
        X, Y = self.track.get_global_position(xcurv_est[0], xcurv_est[1])
        psi = self.track.get_orientation(xcurv_est[0], xcurv_est[1])
        xglob_est = np.zeros((X_DIM,))
        xglob_est[3] = xcurv_est[3]
        xglob_est[2] = psi
        xglob_est[0] = X
        xglob_est[1] = Y
        return xcurv_est, xglob_est

    def get_trajectory_nsteps(self, t0, delta_t, n):
        xcurv_est_nsteps = np.zeros((X_DIM, n))
        xglob_est_nsteps = np.zeros((X_DIM, n))
        for index in range(n):                                                            
            xcurv_est, xglob_est = self.get_estimation(self.time + index * delta_t)      
            xcurv_est_nsteps[:, index] = xcurv_est
            xglob_est_nsteps[:, index] = xglob_est
        return xcurv_est_nsteps, xglob_est_nsteps   #dim=4*n

    def forward_dynamics(self):
        self.time += self.timestep
        self.xcurv, self.xglob = self.get_estimation(self.time)

class DynamicBicycleModel(ModelBase):
    def __init__(self, name=None, param=None, xcurv=None, xglob=None, system_param=None):
        ModelBase.__init__(self, name=name, param=param, system_param=system_param)

    def forward_dynamics(self, realtime_flag):                   
        # This function computes the system evolution. 
        xglob_next = np.zeros((X_DIM,))
        xcurv_next = np.zeros((X_DIM,))

        xglob_next = self.u
        xcurv_next[0],xcurv_next[1],xcurv_next[2],_ = self.track.get_local_position(xglob_next[0],xglob_next[1],xglob_next[2])
        xcurv_next[3] = xglob_next[3]           

        self.xcurv = xcurv_next      
        self.xglob = xglob_next
        #print(self.xglob)
        #print(self.xcurv)

        self.time += self.timestep


# Base Simulator
class CarRacingSim:
    def __init__(self):
        self.track = None                  
        self.track_center = None
        self.vehicles = {}                 
        self.opti_traj_xglob = None        

    def set_timestep(self, dt):
        self.timestep = dt

    def set_track(self, track, track_center):
        self.track = track
        self.track_center = track_center

    def set_opti_traj(self, opti_traj_xglob):
        self.opti_traj_xglob = opti_traj_xglob

