import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anim
from matplotlib import animation
from utils import base, racing_env
from utils.constants import *



class MPCRacingGame(base.MPCRacingGame):
    def __init__(self, mpc_param, racing_game_param=None, system_param=None):
        base.MPCRacingGame.__init__(self, mpc_param, racing_game_param=racing_game_param, system_param=system_param)
        self.realtime_flag = False

# off-board dynamic model
class DynamicBicycleModel(base.DynamicBicycleModel):
    def __init__(self, name=None, param=None, xcurv=None, xglob=None, system_param=None):
        base.DynamicBicycleModel.__init__(self, name=name, param=param, system_param=system_param)

    # in this estimation, the vehicles is assumed to move with input is equal to zero,which means all velocities are konstant
    def get_estimation(self, xglob, xcurv):
        curv = racing_env.get_curvature(self.lap_length, self.point_and_tangent, xcurv[0])
        xcurv_est = np.zeros((X_DIM,))
        xglob_est = np.zeros((X_DIM,))

        # vehicles' current position in global or frenet coordinates
        wz = 0
        xcurv_est[3] = xcurv[3]                          #input=0, v = const.
        xcurv_est[2] = xcurv[2] + self.timestep * (      
            wz - (xcurv[3] * np.cos(xcurv[2]))/ (1 - curv * xcurv[1])* curv
        )  #epsi(k+1)
        xcurv_est[0] = xcurv[0] + self.timestep * (
            (xcurv[3] * np.cos(xcurv[2]) ) / (1 - curv * xcurv[1])
        )  #s(k+1)
        xcurv_est[1] = xcurv[1] + self.timestep * (xcurv[3] * np.sin(xcurv[2]) )#ey(k+1)

        xglob_est[3] = xglob[3]
        xglob_est[2] = xglob[2] + self.timestep * wz                            #psi(k+1)
        xglob_est[0] = xglob[0] + self.timestep * (xglob[3] * np.cos(xglob[2])) #X(k+1)
        xglob_est[1] = xglob[1] + self.timestep * (xglob[3] * np.sin(xglob[2])) #Y(k+1)
             
        return xcurv_est, xglob_est

    # get prediction for se_ibr
    def get_trajectory_nsteps(self, n):    # n = Np+1
        xcurv_nsteps = np.zeros((X_DIM, n))
        xglob_nsteps = np.zeros((X_DIM, n))
        for index in range(n):
            if index == 0:
                xcurv_est, xglob_est = self.get_estimation(self.xglob, self.xcurv)
            else:
                xcurv_est, xglob_est = self.get_estimation(
                    xglob_nsteps[:, index - 1], xcurv_nsteps[:, index - 1]  
                )
            while xcurv_est[0] > self.lap_length:
                xcurv_est[0] = xcurv_est[0] - self.lap_length
            xcurv_nsteps[:, index] = xcurv_est
            xglob_nsteps[:, index] = xglob_est

        return xcurv_nsteps, xglob_nsteps


class NoDynamicsModel(base.NoDynamicsModel):
    def __init__(self, name=None, param=None, xcurv=None, xglob=None):
        base.NoDynamicsModel.__init__(self, name=name, param=param)


# off-board simulator
class CarRacingSim(base.CarRacingSim):
    def __init__(self):
        base.CarRacingSim.__init__(self)
        self.ax = None
        self.fig = None

    def add_vehicle(self, vehicle):
        self.vehicles[vehicle.name] = vehicle                                  
        self.vehicles[vehicle.name].set_track(self.track, self.track_center)   
        self.vehicles[vehicle.name].set_timestep(self.timestep)                

    def sim(
        self,
        sim_time=50.0,
        two_lap=False,
        two_lap_name=None,                                        #"ego"
        animating_flag=False,
    ):

        for i in range(0, int(sim_time / self.timestep)):
            for name in self.vehicles:
                # update system state
                self.vehicles[name].forward_one_step(self.vehicles[name].realtime_flag) 
                #self.calc_ctrl_input(),forward_dynamics(realtime_flag),update_memory()
            if (two_lap == True) and (self.vehicles[two_lap_name].laps > 1):
                print("lap of ego completed")
                break

    def plot_state(self, names):
        fig, axs = plt.subplots(4)             
        for name in names:
            laps = self.vehicles[name].laps  
            time = np.zeros(int(round(self.vehicles[name].time / self.timestep)) + 1) 
            traj = np.zeros((int(round(self.vehicles[name].time / self.timestep)) + 1, X_DIM))
            counter = 0
            for i in range(0, laps):
                for j in range(
                    0,
                    int(
                        round(
                            (self.vehicles[name].times[i][-1] - self.vehicles[name].times[i][0])
                            / self.timestep
                        )
                    ),
                ):
                    time[counter] = self.vehicles[name].times[i][j] 

                    traj[counter, :] = self.vehicles[name].xglobs[i][j][:]
                    counter = counter + 1
            for i in range(
                0,
                int(
                    round(
                        (self.vehicles[name].lap_times[-1] - self.vehicles[name].lap_times[0])
                        / self.timestep
                    )
                )
                + 1, 
            ):
                time[counter] = self.vehicles[name].lap_times[i]
                traj[counter, :] = self.vehicles[name].lap_xglobs[i][:]
                counter = counter + 1
        
            axs[0].plot(time, traj[:, 0], "-o", linewidth=1, markersize=1)
            axs[1].plot(time, traj[:, 1], "-o", linewidth=1, markersize=1)
            axs[2].plot(time, traj[:, 2], "-o", linewidth=1, markersize=1)
            axs[3].plot(time, traj[:, 3], "-o", linewidth=1, markersize=1)

        axs[0].set_xlabel("time [s]", fontsize=14)
        axs[0].set_ylabel("$x$ [m]", fontsize=14)

            
        axs[1].set_xlabel("time [s]", fontsize=14)
        axs[1].set_ylabel("$y$ [m]", fontsize=14)

            
        axs[2].set_xlabel("time [s]", fontsize=14)
        axs[2].set_ylabel("$\psi$ [rad]", fontsize=14)

            
        axs[3].set_xlabel("time [s]", fontsize=14)
        axs[3].set_ylabel("$v$ [m/s]", fontsize=14)
        #plt.show()

    def plot_states(self):
        for name in self.vehicles:
            self.plot_state(name)
        #plt.show()

    def plot_input(self, name):
        laps = self.vehicles[name].laps
        time = np.zeros(int(round(self.vehicles[name].time / self.timestep))) 
        u = np.zeros((int(round(self.vehicles[name].time / self.timestep)), 2))
        counter = 0
        for i in range(0, laps):
            for j in range(
                0,
                int(
                    round(
                        (self.vehicles[name].times[i][-1] - self.vehicles[name].times[i][0])
                        / self.timestep
                    )
                ),
            ):
                time[counter] = self.vehicles[name].times[i][j]
                u[counter, :] = self.vehicles[name].inputs[i][j][:]
                counter = counter + 1
        for i in range(
            0,
            int(
                round(
                    (self.vehicles[name].lap_times[-1] - self.vehicles[name].lap_times[0])
                    / self.timestep
                )
            ),
        ):
            time[counter] = self.vehicles[name].lap_times[i]
            u[counter, :] = self.vehicles[name].lap_inputs[i][:]
            counter = counter + 1
        fig, axs = plt.subplots(2)
        axs[0].plot(time, u[:, 0], "-o", linewidth=1, markersize=1)
        axs[0].set_xlabel("time [s]", fontsize=14)
        axs[0].set_ylabel("$a$ [rad]", fontsize=14)
        axs[1].plot(time, u[:, 1], "-o", linewidth=1, markersize=1)
        axs[1].set_xlabel("time [s]", fontsize=14)
        axs[1].set_ylabel("$/delta$ [m/s^2]", fontsize=14)
        #plt.show()

    def plot_inputs(self):
        for name in self.vehicles:
            self.plot_input(name)
        plt.show()

    def plot_simulation(self):
        fig, ax = plt.subplots()
        # plotting racing track
        self.track.plot_track(ax) 
        # plot trajectories
        for name in self.vehicles:
            laps = self.vehicles[name].laps
            trajglob = np.zeros((int(round(self.vehicles[name].time / self.timestep)) + 1, X_DIM))
            counter = 0
            for i in range(0, laps):
                for j in range(
                    0,
                    int(
                        round(
                            (self.vehicles[name].times[i][-1] - self.vehicles[name].times[i][0])
                            / self.timestep
                        )
                    ),
                ):
                    trajglob[counter, :] = self.vehicles[name].xglobs[i][j][:]
                    counter = counter + 1
            for i in range(
                0,
                int(
                    round(
                        (self.vehicles[name].lap_times[-1] - self.vehicles[name].lap_times[0])
                        / self.timestep
                    )
                )
                + 1,
            ):
                trajglob[counter, :] = self.vehicles[name].lap_xglobs[i][:]
                counter = counter + 1
            ax.plot(trajglob[:, 0], trajglob[:, 1]) 
        #plt.show()

    def animate(
        self, filename="untitled", ani_time=400, lap_number=None, racing_game=False, pillow=False
    ):
        num_veh = len(self.vehicles) - 1
        if racing_game:                                  
            fig = plt.figure(figsize=(10, 4))            
            ax = fig.add_axes([0.05, 0.07, 0.56, 0.9])   
            ax_1 = fig.add_axes([0.63, 0.07, 0.36, 0.9]) 
            ax_1.set_xticks([])                          
            ax_1.set_yticks([])                          
            self.track.plot_track(ax_1, center_line=False)
            patches_vehicles_1 = {}
            patches_vehicles_se_ibr_prediction = []
            patches_vehicles_ibr_prediction = []
            patches_vehicles_mpc_prediction = []
            (se_ibr_prediction_line,) = ax.plot([], [])
            (ibr_prediction_line,) = ax.plot([], [])
            (mpc_prediciton_line,) = ax.plot([], [])
            vehicles_interest = []
            horizon_planner = int(12)  
            se_ibr_prediction = np.zeros((ani_time, horizon_planner + 1, X_DIM))
            ibr_prediction = np.zeros((ani_time, horizon_planner + 1, X_DIM))
            mpc_prediction = np.zeros((ani_time, horizon_planner+1, X_DIM))

        else:
            fig, ax = plt.subplots()
        # plotting racing track
        self.track.plot_track(ax, center_line=False)
        # plot vehicles
        vertex_directions = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
        patches_vehicles = {}
        trajglobs = {}
        lap_number = self.vehicles["ego"].laps     
        time =0
        for i in range(0, lap_number):
            time = time + int(
                round(
                    (self.vehicles["ego"].times[i][-1] - self.vehicles["ego"].times[i][0])
                    / self.timestep
                )
            )
        
        time = time + int(
            round(
                (self.vehicles["ego"].lap_times[-1] - self.vehicles["ego"].lap_times[0])
                / self.timestep
                )
                )#+ 1
        sim_time = (time)
        print(time)
        
        if ani_time > sim_time:
            ani_time = sim_time
        for name in self.vehicles:
            if name == "ego":
                face_color = "red"
            else:
                face_color = "blue"
            edge_color = "None"
            patches_vehicle = patches.Polygon(
                vertex_directions,
                alpha=1.0,      
                closed=True,    
                fc=face_color,
                ec="None",     
                zorder=10,      
                linewidth=2,
            )
            if racing_game:
                patches_vehicle_1 = patches.Polygon(            
                    vertex_directions,
                    alpha=1.0,
                    closed=True,
                    fc=face_color,
                    ec="None",
                    zorder=10,
                    linewidth=2,
                )
                if name == "ego":
                    for kkkk in range(0, 6+ 1):               
                        patch_se_ibr = patches.Polygon(       
                            vertex_directions,
                            alpha=1.0 - kkkk * 0.15,
                            closed=True,
                            fc="None",
                            zorder=10,
                            linewidth=2,
                        )
                        patches_vehicles_se_ibr_prediction.append(patch_se_ibr)
                        ax.add_patch(patches_vehicles_se_ibr_prediction[kkkk])                        
                else:
                    for jjjj in range(0, 6 + 1):                
                        patch_mpc = patches.Polygon(           
                            vertex_directions,
                            alpha=1.0 - jjjj * 0.15,
                            closed=True,
                            fc="None",
                            zorder=10,
                            linewidth=2,
                        )
                        patches_vehicles_mpc_prediction.append(patch_mpc)
                        ax.add_patch(patches_vehicles_mpc_prediction[jjjj]) 

                    for kkkk in range(0, 6 + 1):               
                        patch_ibr = patches.Polygon(       
                            vertex_directions,
                            alpha=1.0 - kkkk * 0.15,
                            closed=True,
                            fc="None",
                            zorder=10,
                            linewidth=2,
                        )
                        patches_vehicles_ibr_prediction.append(patch_ibr)
                        ax.add_patch(patches_vehicles_ibr_prediction[kkkk]) 

            if name == "ego":
                if racing_game:
                    pass
                else:
                    ax.add_patch(patches_vehicle)  
            else:
                ax.add_patch(patches_vehicle)
            
            if racing_game:
                ax_1.add_patch(patches_vehicle_1)
                ax_1.axis("equal") 
                ax.add_line(se_ibr_prediction_line)
                ax.add_line(ibr_prediction_line)
                ax.add_line(mpc_prediciton_line)               
                patches_vehicles_1[name] = patches_vehicle_1
            
            ax.axis("equal")                     
            patches_vehicles[name] = patches_vehicle
            counter = 0
            trajglob = np.zeros((ani_time, X_DIM))
            for j in range(ani_time):
                trajglob[counter, :] = self.vehicles[name].xglob_log[j][:]               
                if racing_game:
                    if name == "ego":
                        se_ibr_prediction[counter, :, :] = self.vehicles[
                            name
                        ].se_ibr_prediction[j][:, :]                        

                        if self.vehicles[name].vehicles_interest[-1-j] is None: 
                            vehicles_interest.insert(0, None)  
                        else:
                            vehicles_interest.insert(
                                0,
                                self.vehicles[name].vehicles_interest[-1-j],
                            )
                    else:
                        if name == "mpc":
                            mpc_prediction[counter, :, :] = self.vehicles[
                                name
                            ].mpc_prediction[j][:, :]
                        if name == "ibr":
                            ibr_prediction[counter, :, :] = self.vehicles[
                                name
                            ].ibr_prediction[j][:, :]                   
                counter = counter + 1
            trajglobs[name] = trajglob

        def update(i):  
            if racing_game:
                ax_1.set_xlim([trajglobs["ego"][i , 0] - 2, trajglobs["ego"][i , 0] + 2]) 
                ax_1.set_ylim([trajglobs["ego"][i , 1] - 2, trajglobs["ego"][i , 1] + 2])
            for name in patches_vehicles:
                x, y = trajglobs[name][i, 0], trajglobs[name][i, 1] 
                psi = trajglobs[name][i, 2]
                l = self.vehicles[name].param.length / 2
                w = self.vehicles[name].param.width / 2
                vertex_x = [                                
                    x + l * np.cos(psi) - w * np.sin(psi),
                    x + l * np.cos(psi) + w * np.sin(psi),
                    x - l * np.cos(psi) + w * np.sin(psi),
                    x - l * np.cos(psi) - w * np.sin(psi),
                ]
                vertex_y = [
                    y + l * np.sin(psi) + w * np.cos(psi),
                    y + l * np.sin(psi) - w * np.cos(psi),
                    y - l * np.sin(psi) - w * np.cos(psi),
                    y - l * np.sin(psi) + w * np.cos(psi),
                ]
                patches_vehicles[name].set_xy(np.array([vertex_x, vertex_y]).T) 
                if racing_game:
                    patches_vehicles_1[name].set_xy(np.array([vertex_x, vertex_y]).T)
                    if name == "ego":
                        patches_vehicles[name].set_facecolor("None")
                        if se_ibr_prediction[i, :, :].all == 0: 
                            for jjj in range(0, 6 ):
                                patches_vehicles_se_ibr_prediction[jjj].set_facecolor("None")
                        else:
                            for iii in range(0, 6):
                                x, y = (
                                    se_ibr_prediction[i, iii * 2, 0],
                                    se_ibr_prediction[i, iii * 2, 1],
                                )
                                if x == 0.0 and y == 0.0:
                                    patches_vehicles_se_ibr_prediction[iii].set_facecolor("None")
                                else:
                                    patches_vehicles_se_ibr_prediction[iii].set_facecolor("red")
                                psi = se_ibr_prediction[i, iii , 3]
                                vertex_x = [
                                    x + l * np.cos(psi) - w * np.sin(psi),
                                    x + l * np.cos(psi) + w * np.sin(psi),
                                    x - l * np.cos(psi) + w * np.sin(psi),
                                    x - l * np.cos(psi) - w * np.sin(psi),
                                ]
                                vertex_y = [
                                    y + l * np.sin(psi) + w * np.cos(psi),
                                    y + l * np.sin(psi) - w * np.cos(psi),
                                    y - l * np.sin(psi) - w * np.cos(psi),
                                    y - l * np.sin(psi) + w * np.cos(psi),
                                ]
                                patches_vehicles_se_ibr_prediction[iii].set_xy(
                                    np.array([vertex_x, vertex_y]).T
                                )
                    else:
                        patches_vehicles[name].set_facecolor("None")
                        if ibr_prediction[i, :, :].all == 0:
                            for jjj in range(0, 6):
                                patches_vehicles_ibr_prediction[jjj].set_facecolor("None")
                        else:
                            for iii in range(0, 6):
                                x, y = (
                                    ibr_prediction[i, iii * 2, 0],
                                    ibr_prediction[i, iii * 2, 1],
                                )
                                if x == 0.0 and y == 0.0:
                                    patches_vehicles_ibr_prediction[iii].set_facecolor("None")
                                else:
                                    patches_vehicles_ibr_prediction[iii].set_facecolor("yellow")
                                psi = ibr_prediction[i , iii, 3]
                                vertex_x = [
                                    x + l * np.cos(psi) - w * np.sin(psi),
                                    x + l * np.cos(psi) + w * np.sin(psi),
                                    x - l * np.cos(psi) + w * np.sin(psi),
                                    x - l * np.cos(psi) - w * np.sin(psi),
                                ]
                                vertex_y = [
                                    y + l * np.sin(psi) + w * np.cos(psi),
                                    y + l * np.sin(psi) - w * np.cos(psi),
                                    y - l * np.sin(psi) - w * np.cos(psi),
                                    y - l * np.sin(psi) + w * np.cos(psi),
                                ]
                                patches_vehicles_ibr_prediction[iii].set_xy(
                                    np.array([vertex_x, vertex_y]).T
                                )
                        
                        if mpc_prediction[i, :, :].all == 0: 
                            for iii in range(0, 6):
                                patches_vehicles_mpc_prediction[iii].set_facecolor("None")
                        else: 
                            for jjj in range(0, 6):
                                x, y = (
                                    mpc_prediction[i, jjj * 2, 0], 
                                    mpc_prediction[i, jjj * 2, 1],
                                )
                                if x == 0 and y == 0:
                                    patches_vehicles_mpc_prediction[jjj].set_facecolor("None")
                                else:
                                    patches_vehicles_mpc_prediction[jjj].set_facecolor("blue")
                                psi = mpc_prediction[i, jjj, 2]
                                vertex_x = [
                                    x + l * np.cos(psi) - w * np.sin(psi),
                                    x + l * np.cos(psi) + w * np.sin(psi),
                                    x - l * np.cos(psi) + w * np.sin(psi),
                                    x - l * np.cos(psi) - w * np.sin(psi),
                                ]
                                vertex_y = [
                                    y + l * np.sin(psi) + w * np.cos(psi),
                                    y + l * np.sin(psi) - w * np.cos(psi),
                                    y - l * np.sin(psi) - w * np.cos(psi),
                                    y - l * np.sin(psi) + w * np.cos(psi),
                                ]
                                patches_vehicles_mpc_prediction[jjj].set_xy(
                                    np.array([vertex_x, vertex_y]).T
                                )
                    
                    if se_ibr_prediction[i, :, :].all == 0:
                        se_ibr_prediction_line.set_data([], [])
                    else:
                        se_ibr_prediction_line.set_data(
                            se_ibr_prediction[i, :, 0], se_ibr_prediction[i, :, 1]
                        )
                        se_ibr_prediction_line.set_color("red")
                        se_ibr_prediction_line.set_linewidth(2)

                    if ibr_prediction[i, :, :].all == 0:
                        ibr_prediction_line.set_data([], [])
                    else:
                        ibr_prediction_line.set_data(
                            ibr_prediction[i, :, 0], ibr_prediction[i, :, 1]
                        )
                        ibr_prediction_line.set_color("orange")
                        ibr_prediction_line.set_linewidth(2)   

                    if mpc_prediction[i, :, :].all == 0:
                        mpc_prediciton_line.set_data([], [])
                    else:
                        mpc_prediciton_line.set_data(
                            mpc_prediction[i, :, 0], mpc_prediction[i, :, 1]
                        )
                        mpc_prediciton_line.set_color("green")
                        mpc_prediciton_line.set_linewidth(2)
                    
                    if vehicles_interest == []:              
                        pass
                    else:
                        if vehicles_interest[i] is None:
                            if name == "ego":
                                patches_vehicles[name].set_facecolor("None")
                                patches_vehicles_1[name].set_facecolor("red")
                            else:
                                patches_vehicles[name].set_facecolor("blue")
                                patches_vehicles_1[name].set_facecolor("blue")
                        else:
                            veh_of_interest = False
                            for name_1 in list(vehicles_interest[i]):
                                if name == name_1:
                                    veh_of_interest = True
                            if veh_of_interest:
                                patches_vehicles[name].set_facecolor("green")
                                patches_vehicles_1[name].set_facecolor("green")
                            else:
                                if name == "ego":
                                    patches_vehicles[name].set_facecolor("None")
                                    patches_vehicles_1[name].set_facecolor("red")
                                else:
                                    patches_vehicles[name].set_facecolor("blue")
                                    patches_vehicles_1[name].set_facecolor("blue")

        media = anim.FuncAnimation(
            fig, update, frames=np.arange(0, trajglob.shape[0]), interval=100
        ) 
        if pillow:
            media.save(
                "media/animation/" + filename + ".gif",
                dpi=80,  
                writer="pillow",
            )
        else:
            media.save(
                "media/animation/" + filename + ".gif",
                dpi=80,
                writer=animation.writers["ffmpeg"](fps=10),  # fps: movie frame rate (per second)
            )
