import numpy as np


def vehicle_dynamics(dynamics_param, curv, xglob, xcurv, delta_t, u):

    m, lf, lr, Iz, Df, Cf, Bf, Dr, Cr, Br, L= dynamics_param.get_params()

    xglob_next = np.zeros(len(xglob))   # dim = 4
    xcurv_next = np.zeros(len(xcurv))   # dim = 4

    a = u[0]                            # acceleration at vehicle's center of gravity    
    delta = u[1]                        # steering angle    
    
    v = xglob[3]                        # longitudinal velocity
    psi = xglob[2]                      # heading angle of the vehicle
    X = xglob[0]                        # inertial coordinates of the location of the c.g. of the vehicle
    Y = xglob[1]


    epsi = xcurv[2]                     # heading angle error between vehicle and center line
    s = xcurv[0]                        # curvilinear distance travelled along the track's center line
    ey = xcurv[1]                       # deviation distance between vehicle and center line
    
    # Propagate the dynamics of delta_t
    xglob_next[0] = X + delta_t * (v * np.cos(psi))                                
    xglob_next[1] = Y + delta_t * (v * np.sin(psi))
    xglob_next[2] = psi + delta_t * (v * np.tan(delta) / L)
    xglob_next[3] = v + delta_t * a
    
    wz = v * np.tan(delta) / L

    xcurv_next[0] = s + delta_t * (v * np.cos(epsi)  / (1 - curv * ey)) 
    xcurv_next[1] = ey + delta_t * (v * np.sin(epsi))
    xcurv_next[2] = epsi + delta_t * (wz - v * np.cos(epsi)/ (1 - curv * ey) * curv)
    xcurv_next[3] = v + delta_t * (a )

    return xglob_next, xcurv_next
