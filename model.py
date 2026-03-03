import numpy as np
import control as ct

#constants
nonlinear_params = {
    'm' : 0.8,
    'n' : 4,
    'K_phi2': 16.5, #h^-1
    'K_phi1': -0.1116, #(Tons * h)^-1
    'T_f' : 0.3, #h
    'T_r' : 0.01, #h
    'd' : 1,
    }


#functions
# https://python-control.readthedocs.io/en/0.10.2/nonlinear.html 
def nonlinear_update(t, x, u, params):

    y_f, z, y_r, u, v, sigma_yf, sigma_z = x
    du, dv = u
    m, n, K_phi2, K_phi1, T_f, T_r, d = map(params.get, ['m', 'n', 'K_phi2', 'K_phi1', 'T_f', 'T_r', 'd'])

    phi = max(0, (-d*K_phi1*z**2 + K_phi2*z))
    K_alpha = 570**m * 170**n *((570 / 450) - 1) 
    alpha = (phi**m * v**n)/(K_alpha + phi**m * v**n)

    y_fdot = (-y_f + (1-alpha)*phi)/T_f
    zdot = -phi + u + y_r
    y_rdot = (-y_r + alpha*phi)/T_r

    udot = du
    vdot = dv

    sigma_yfdot = 0 #adjust later
    sigma_zdot = 0
    
    return np.array([y_fdot, zdot, y_rdot, udot, vdot, sigma_yfdot, sigma_zdot])

def nonlinear_output(t, x, u, params):
    return np.array(x[:3])

def simulate_nonlinear(params):
    mill_nonlinear = ct.nlsys(
    nonlinear_update, nonlinear_output, name='mill_nonlinear',
    params=params, states=['y_f', 'z', 'y_r', 'u', 'v', 'sigma_yf', 'sigma_z'],
    outputs=['y_f', 'z', 'y_r'], inputs=['du', 'dv'])
    



if __name__ == '__main__':
    simulate_nonlinear(nonlinear_params)