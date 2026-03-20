import numpy as np
import control as ct
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy.linalg as la
import cvxpy as cp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Tunable parameters
alpha=10
initial_guesses = [100.0, 120.0, 100.0] #120 for equi points, but other 2 are random


# Constants
nonlinear_params = {
    'm' : 0.8,
    'n' : 4,
    'K_phi2': 16.5, #h^-1
    'K_phi1': -0.1116, #(Tons * h)^-1
    'T_f' : 0.3, #h
    'T_r' : 0.01, #h
    'd' : 1,
    'yf_equi':120, #tons/h
    'z_equi':75, #tons
    }


mpc_params = {
    'Q': np.eye(7),
    'R': np.eye(2) * 0.001,
    #from reference: *0.001, maar zij hebben een andere terminal set constraint (lees even dit van chat):

    #they define their terminal set $X(K(x^*))$ as any state where the linear LQR controller can take over and safely drive the system to equilibrium without violating constraints.
    #They also approximate their terminal cost by artificially simulating the linear controller forward for $M$ extra steps inside the optimization problem.

    #wij mogen dit niet doen, dus heb R even groter gemaakt. 

    'N': 60  
}

# Constraints
x_min_absolute = np.array([
    -np.inf,  # y_f no limit
    50.0,     # z: minimum safe mill load (Tons)
    -np.inf,  # y_r: no limit
    0.0,      # u: cannot be negative
    100.0,    # v: minimum speed (r/min)
    -np.inf,  
    -np.inf   
])

x_max_absolute = np.array([
    np.inf,  
    100.0,    
    np.inf,   
    200.0,    
    250.0,    
    np.inf,   
    np.inf   
])

#dit is gokje, die reference papier gebruikt dit ook, maar zegt nergens wat de value is vgm, dus dacht zet er even 10 bij
U_min = np.array([
    -10.0,    
    -10.0     
])

U_max = np.array([
    10.0,     
    10.0     
])


####################### nonlinear model ######################################################
#functions
# https://python-control.readthedocs.io/en/0.10.2/nonlinear.html 
def nonlinear_update(t, x, U, params):
    '''
    Applies the update step of the nonlinear system
    '''
    y_f, z, y_r, u, v, sigma_yf, sigma_z = x
    du, dv = U
    m, n, K_phi2, K_phi1, T_f, T_r, d = map(params.get, ['m', 'n', 'K_phi2', 'K_phi1', 'T_f', 'T_r', 'd'])

    phi = max(0, (-d*K_phi1*z**2 + K_phi2*z))
    K_alpha = 570**m * 170**n *((570 / 450) - 1) 
    alpha = (phi**m * v**n)/(K_alpha + phi**m * v**n)

    y_fdot = (-y_f + (1-alpha)*phi)/T_f
    zdot = -phi + u + y_r
    y_rdot = (-y_r + alpha*phi)/T_r

    udot = du
    vdot = dv

    yf_equi, z_equi = params['yf_equi'], params['z_equi']
    sigma_yfdot = y_f - yf_equi
    sigma_zdot = z-z_equi
    
    return np.array([y_fdot, zdot, y_rdot, udot, vdot, sigma_yfdot, sigma_zdot])

def nonlinear_output(t, x, U, params):
    '''
    Applies the output step of the nonlinear sysem
    '''
    return np.array(x[:3])

def simulate_nonlinear(params, t, U):
    '''
    Simulates the nonlinear system with given parameters and inputs
    '''
    mill_nonlinear = ct.nlsys(
    nonlinear_update, nonlinear_output, name='mill_nonlinear',
    params=params, states=['y_f', 'z', 'y_r', 'u', 'v', 'sigma_yf', 'sigma_z'],
    outputs=['y_f', 'z', 'y_r'], inputs=['du', 'dv'])

    resp = ct.input_output_response(mill_nonlinear, t, U)
    resp.plot()
    plt.show()


######################## Find Equilibrium ##########################################
def find_equi(params, initial_guesses):
    '''
    Identifies the equilibrium based on a given inital guess and the yf and z equilibria in the parameters
    '''
    def test_guess(unknowns):
        
        y_r, u, v=unknowns
        y_f=params['yf_equi']
        z=params['z_equi']

        #[y_f, z, y_r, u, v, sigma_yf, sigma_z]
        current_x = [y_f,z,y_r,u, v,0,0]
        current_U=[0,0]

        derivatives = nonlinear_update(0,current_x, current_U, params)

        y_fdot = derivatives[0]
        zdot = derivatives[1]
        y_rdot = derivatives[2]

        return [y_fdot, zdot, y_rdot]

    
    answer = fsolve(test_guess, initial_guesses)

    yr_final, u_final, v_final = answer

    x_equi= np.array([params['yf_equi'], params['z_equi'], yr_final, u_final, v_final, 0.0, 0.0])
    U_equi = np.array([0.0, 0.0])

    return x_equi, U_equi


######################## Linearize ####################################
def linearize_sys(x_equi, U_equi, params, delta):
    '''
    Linearizes the system around the equilibrium
    '''
    num_states = len(x_equi)
    num_inputs = len(U_equi)

    A = np.zeros((num_states, num_states))
    B = np.zeros((num_states, num_inputs))

    # A
    for i in range(num_states):
        x_plus = np.copy(x_equi)
        x_minus = np.copy(x_equi)

        x_plus[i] = x_plus[i] + delta
        x_minus[i] = x_minus[i] - delta

        #delta
        f_plus = nonlinear_update(0, x_plus, U_equi, params)
        f_minus = nonlinear_update(0, x_minus, U_equi, params)

        #slope
        A[:, i] = (f_plus - f_minus) / (2 * delta)

    #B
    for j in range(num_inputs):
        U_plus = np.copy(U_equi)
        U_minus = np.copy(U_equi)

        U_plus[j] = U_plus[j] + delta
        U_minus[j] = U_minus[j] - delta

        f_plus = nonlinear_update(0, x_equi, U_plus, params)
        f_minus = nonlinear_update(0, x_equi, U_minus, params)

        B[:, j] = (f_plus - f_minus) / (2 * delta)

    return A, B

########################## discretize #####################################

def discretize_sys(A_cont, B_cont, Ts):
    '''
    Discretized the update system for a given Ts
    '''
    Ts_h = Ts/60

    C = np.zeros((2, 7))
    D = np.zeros((2, 2))

    sys_cont = ct.StateSpace(A_cont, B_cont, C, D)

    sys_disc = ct.c2d(sys_cont, Ts_h, method='zoh')

    A_disc = sys_disc.A
    B_disc = sys_disc.B

    return A_disc, B_disc


################################ terminal matrices and such###############################

def terminal_matrices(A_disc, B_disc, Q, R):
    '''
    Generate terminal matrices
    '''
    K,P,eigenvalues = ct.dlqr(A_disc, B_disc, Q, R)

    return K, P


def check_terminal(P, K, alpha, x_min_dev, x_max_dev, U_min, U_max):
    '''
    Checks whether the terminal matrices satisfy the constraints
    '''
    P_inv = la.inv(P)

    print(f"checking alpha={alpha}")

    state_safe=True
    input_safe = True 

    for i in range(len(P)):
        max_reach = np.sqrt(alpha*P_inv[i,i])

        if max_reach > x_max_dev[i] or -max_reach < x_min_dev[i]:
            state_safe=False
            print("state exceeds limit")
    
    U_shape = K @ P_inv @ K.T

    for j in range(len(U_shape)):
        max_u_reach = np.sqrt(alpha * U_shape[j, j])

        if max_u_reach > U_max[j] or -max_u_reach < U_min[j]:
            input_safe=False
            print("input exceeds limit")

    if state_safe and input_safe:
        print("alpha good")
    else:
        print("alpha bad")

################ superior alpha finding (chatje) ##################################
def find_max_alpha(P, K, x_min_dev, x_max_dev, U_min, U_max):
    """
    Calculates the maximum mathematically valid alpha for the terminal 
    invariant set based on state and input constraints.
    """
    P_inv = la.inv(P)
    U_shape = K @ P_inv @ K.T
    
    alpha_max = np.inf
    
    # 1. Find alpha limits based on State Constraints
    for i in range(len(x_max_dev)):
        p_inv_ii = P_inv[i, i]
        if p_inv_ii > 1e-10: # Prevent division by zero
            # Check upper bounds
            if x_max_dev[i] < np.inf:
                alpha_limit = (x_max_dev[i]**2) / p_inv_ii
                alpha_max = min(alpha_max, alpha_limit)
            
            # Check lower bounds (squaring removes the negative sign)
            if x_min_dev[i] > -np.inf:
                alpha_limit = (x_min_dev[i]**2) / p_inv_ii
                alpha_max = min(alpha_max, alpha_limit)
                
    # 2. Find alpha limits based on Input Constraints (-Kx)
    for j in range(len(U_max)):
        u_shape_jj = U_shape[j, j]
        if u_shape_jj > 1e-10:
            # Check upper bounds
            if U_max[j] < np.inf:
                alpha_limit = (U_max[j]**2) / u_shape_jj
                alpha_max = min(alpha_max, alpha_limit)
            
            # Check lower bounds
            if U_min[j] > -np.inf:
                alpha_limit = (U_min[j]**2) / u_shape_jj
                alpha_max = min(alpha_max, alpha_limit)
                
    # Multiply by 0.99 to give the solver a tiny 1% safety margin
    # against floating-point rounding errors during optimization
    safe_alpha = alpha_max * 0.99 
    
    print(f"Calculated Maximum Safe Alpha: {safe_alpha:.4f}")
    return safe_alpha

#################### MPC'tje ############################
def setup_mpc_problem(A_disc, B_disc, Q, R, P, alpha, x_min_dev, x_max_dev, U_min, U_max, N):
    '''
    Defines the problem in a way that's more easily usable in python
    '''
    n_states = A_disc.shape[0]
    n_inputs = B_disc.shape[1]

    x = cp.Variable((n_states, N + 1))
    u = cp.Variable((n_inputs, N))

    x_init = cp.Parameter(n_states)

    cost = 0
    constraints = []

    # inital coniditions
    constraints += [x[:, 0] == x_init]

    # loop over horizon
    for k in range(N):
        # stage cost
        cost += cp.quad_form(x[:, k], Q) + cp.quad_form(u[:, k], R) 

        # dyn. const. 
        constraints += [x[:, k+1] == A_disc @ x[:, k] + B_disc @ u[:, k]]

        # phys. const.
        constraints += [x_min_dev <= x[:, k], x[:, k] <= x_max_dev]
        constraints += [U_min <= u[:, k], u[:, k] <= U_max]


    # terminal costs
    cost += cp.quad_form(x[:, N], P)

    # terminal state constraint
    constraints += [x_min_dev <= x[:, N], x[:, N] <= x_max_dev]
    
    #terminal invariant set constraint
    constraints += [cp.quad_form(x[:, N], P) <= alpha]

    prob = cp.Problem(cp.Minimize(cost), constraints)

    return prob, x_init, x, u


if __name__ == '__main__':
    t = np.linspace(0, 500)
    U = np.ones([2, t.size])
    #simulate_nonlinear(nonlinear_params, t, U)

    #check equi
    x_equi, U_equi = find_equi(nonlinear_params, initial_guesses) 
    print(np.round(x_equi, 2))
    check_derivatives = nonlinear_update(0, x_equi, U_equi, nonlinear_params)
    print(check_derivatives[:3])

    #check linearization 
    A_cont, B_cont = linearize_sys(x_equi, U_equi, nonlinear_params, 0.01)
    print(A_cont)
    print(B_cont)

    #test disc
    A_disc, B_disc = discretize_sys(A_cont, B_cont, 1 )
    print(A_disc)
    print(B_disc)

    # testing alpha
    x_min_dev = x_min_absolute - x_equi
    x_max_dev = x_max_absolute - x_equi
    
    Q = mpc_params['Q']
    R = mpc_params['R']
    N = mpc_params['N']

    K, P = terminal_matrices(A_disc, B_disc, Q, R)

    # testing superior alph finding
    alpha = find_max_alpha(P, K, x_min_dev, x_max_dev, U_min, U_max) 
    check_terminal(P, K, alpha, x_min_dev, x_max_dev, U_min, U_max)


    # testing optimisation, want parameters waren kut

    # initialize problem
    prob, x_init_param, x_var, u_var = setup_mpc_problem(
        A_disc, B_disc, Q, R, P, alpha, 
        x_min_dev, x_max_dev, U_min, U_max, N
    )
    # initial disturbance 
    x_test_state = np.zeros(7) # start at 0
    x_test_state[1] = 1     # Add 1 ton
    x_init_param.value = x_test_state
    prob.solve()
    print("solver status:",prob.status)


    ############### de episch MPC loop #########################

    T_sim_minutes = 60  
    Ts = 1.0            
    Ts_h = Ts / 60.0    
    N_steps = int(T_sim_minutes / Ts)

    x_history = np.zeros((7, N_steps + 1))
    u_history = np.zeros((2, N_steps))
    t_history = np.arange(0, T_sim_minutes + Ts, Ts)

    x_abs_current = np.copy(x_equi)
    x_abs_current[1] += 1

    x_history[:, 0] = x_abs_current

    for k in range(N_steps):
        # Convert absolute state measurement to deviation variable
        x_dev_current = x_abs_current - x_equi
    
        x_init_param.value = x_dev_current
        prob.solve() 

        if prob.status != 'optimal':
            print(prob.status)
            break 
        
        # find first control move
        u_dev_optimal = u_var.value[:, 0]
    
        # Convert deviation control back to absolute control 
        U_abs_applied = u_dev_optimal + U_equi
    
        # store the input we are applying
        u_history[:, k] = U_abs_applied
    
        # sim plant
        def plant_dynamics(t, x_state):
            return nonlinear_update(t, x_state, U_abs_applied, nonlinear_params)
    
        sol = solve_ivp(
            plant_dynamics, 
            [0, Ts_h],             
            x_abs_current
        )
    
        #update staet
        x_abs_current = sol.y[:, -1]
        x_history[:, k+1] = x_abs_current


    ################### LQR reference / comparison ####################################


    # Arrays to store LQR data
    x_history_lqr = np.zeros((7, N_steps + 1))
    u_history_lqr = np.zeros((2, N_steps))

    # Use the exact same starting disturbance as the MPC
    x_abs_current_lqr = np.copy(x_equi)
    x_abs_current_lqr[1] += 1 
    x_history_lqr[:, 0] = x_abs_current_lqr
    for k in range(N_steps):
        x_dev_current = x_abs_current_lqr - x_equi
    
        # u = -Kx
        u_dev_lqr = -K @ x_dev_current
    
        U_abs_lqr = u_dev_lqr + U_equi
    
        # saturate inputs instead of constraints
        U_abs_applied = np.clip(U_abs_lqr, U_equi + U_min, U_equi + U_max)
        u_history_lqr[:, k] = U_abs_applied

        # sim plant
        def plant_dynamics_lqr(t, x_state):
            return nonlinear_update(t, x_state, U_abs_applied, nonlinear_params)

        sol = solve_ivp(
            plant_dynamics_lqr, 
            [0, Ts_h],             
            x_abs_current_lqr
        )

        #update state
        x_abs_current_lqr = sol.y[:, -1]
        x_history_lqr[:, k+1] = x_abs_current_lqr


    ################### plots (chatje) ##################### 
    # --- 2. Plotting Both Controllers ---
    # Create a figure with 5 subplots
    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

    # Time vectors
    t_inputs = t_history[:-1]

    # --- Output States ---
    state_indices = [0, 1, 2]
    state_labels = ['y_f (Product Flow Rate) [Tons/h]', 'z (Mill Load) [Tons]', 'y_r (Tailings Flow Rate) [Tons/h]']

    for i, idx in enumerate(state_indices):
        # Plot MPC (Solid Blue)
        axs[i].plot(t_history, x_history[idx, :], label='MPC', color='blue', linewidth=2)
        # Plot LQR (Dotted Orange)
        axs[i].plot(t_history, x_history_lqr[idx, :], label='LQR Baseline', color='darkorange', linestyle=':', linewidth=2.5)
        # Plot Equilibrium (Dashed Red)
        axs[i].axhline(y=x_equi[idx], color='red', linestyle='--', label='Equilibrium')

        axs[i].set_ylabel(state_labels[i].split(' ')[-1]) # Just grab the units
        axs[i].set_title(state_labels[i])
        axs[i].legend()
        axs[i].grid(True)

    # --- Control Inputs ---
    input_indices = [0, 1]
    input_labels = ['u (Feed Flow Rate) [Tons/h]', 'v (Classifier Speed) [r/min]']

    for j, idx in enumerate(input_indices):
        # Plot MPC (Solid Green)
        axs[j+3].step(t_inputs, u_history[idx, :], label='MPC', color='green', where='post', linewidth=2)
        # Plot LQR (Dotted Purple)
        axs[j+3].step(t_inputs, u_history_lqr[idx, :], label='LQR Baseline', color='purple', linestyle=':', where='post', linewidth=2.5)
        # Plot Equilibrium (Dashed Red)
        axs[j+3].axhline(y=U_equi[idx], color='red', linestyle='--', label='Equilibrium')

        axs[j+3].set_ylabel(input_labels[j].split(' ')[-1])
        axs[j+3].set_title(input_labels[j])
        axs[j+3].legend()
        axs[j+3].grid(True)

    axs[4].set_xlabel('Time (minutes)')
    plt.tight_layout()
    plt.show()