import numpy as np
import numba

from ...utils import model_utils as mu


def timeIntegration(params):
    """Sets up the parameters for time integration

    :param params: Parameter dictionary of the model
    :type params: dict
    :return: Integrated activity variables of the model
    :rtype: (numpy.ndarray,)
    """
    dt = params["dt"]  # Time step for the Euler intergration (ms)
    duration = params["duration"]  # imulation duration (ms)
    RNGseed = params["seed"]  # seed for RNG
    
    # ------------------------------------------------------------------------
    # local parameters
    A = params["A"]
    a = params["a"]
    B = params["B"]
    b = params["b"]
    C = params["C"]
    C1 = params["C1"]
    C2 = params["C2"]
    C3 = params["C3"]
    C4 = params["C4"]
    e0 = params["e0"]
    v0 = params["v0"]
    r = params["r"]
    tau_y3 = params["tau_y3"]
    tau_y4 = params["tau_y4"]  
    tau_y5 = params["tau_y5"]  

    # external input parameters:
    # Parameter of the Ornstein-Uhlenbeck process for the external input(ms)
    tau_ou = params["tau_ou"]
    # Parameter of the Ornstein-Uhlenbeck (OU) process for the external input ( mV/ms/sqrt(ms) )
    sigma_ou = params["sigma_ou"]
    # Mean external excitatory input (OU process)
    y3_ou_mean = params["y3_ou_mean"]
    # Mean external excitatory input (OU process)
    y4_ou_mean = params["y4_ou_mean"]
    # Mean external inhibitory input (OU process)
    y5_ou_mean = params["y5_ou_mean"]

    # ------------------------------------------------------------------------
    # global coupling parameters

    # Connectivity matrix
    # Interareal relative coupling strengths (values between 0 and 1), Cmat(i,j) connection from jth to ith
    Cmat = params["Cmat"]
    N = len(Cmat)  # Number of nodes
    K_gl = params["K_gl"]  # global coupling strength
    # Interareal connection delay
    lengthMat = params["lengthMat"]
    signalV = params["signalV"]

    if N == 1:
        Dmat = np.zeros((N, N))
    else:
        # Interareal connection delays, Dmat(i,j) Connnection from jth node to ith (ms)
        Dmat = mu.computeDelayMatrix(lengthMat, signalV)
        Dmat[np.eye(len(Dmat)) == 1] = np.zeros(len(Dmat))
    Dmat_ndt = np.around(Dmat / dt).astype(int)  # delay matrix in multiples of dt
    
    # ------------------------------------------------------------------------
    # Initialization
    # Floating point issue in np.arange() workaraound: use integers in np.arange()
    t = np.arange(1, round(duration, 6) / dt + 1) * dt  # Time variable (ms)

    sqrt_dt = np.sqrt(dt)

    max_global_delay = np.max(Dmat_ndt)
    startind = int(max_global_delay + 1)  # timestep to start integration at

    # noise variable
    y3_ou = params["y3_ou"].copy()
    y4_ou = params["y4_ou"].copy()
    y5_ou = params["y5_ou"].copy()

    # state variable arrays, have length of t + startind
    # they store initial conditions AND simulated data
    y0s = np.zeros((N, startind + len(t)))
    y1s = np.zeros((N, startind + len(t)))
    y2s = np.zeros((N, startind + len(t)))
    y3s = np.zeros((N, startind + len(t)))
    y4s = np.zeros((N, startind + len(t)))
    y5s = np.zeros((N, startind + len(t)))

    # External input param
    ext_input_static = mu.adjustArrayShape(params["ext_input_static"], y4s)
    ext_input = mu.adjustArrayShape(params["ext_input"], y4s)

    # ------------------------------------------------------------------------
    # Set initial values
    # if initial values are just a Nx1 array
    if np.shape(params["y0_init"])[1] == 1:
        y0_init = np.dot(params["y0_init"], np.ones((1, startind)))
        y1_init = np.dot(params["y1_init"], np.ones((1, startind)))
        y2_init = np.dot(params["y2_init"], np.ones((1, startind)))
        y3_init = np.dot(params["y3_init"], np.ones((1, startind)))
        y4_init = np.dot(params["y4_init"], np.ones((1, startind)))
        y5_init = np.dot(params["y5_init"], np.ones((1, startind)))
    # if initial values are a Nxt array
    else:
        y0_init = params["y0_init"][:, -startind:]
        y1_init = params["y1_init"][:, -startind:]
        y2_init = params["y2_init"][:, -startind:]
        y3_init = params["y3_init"][:, -startind:]
        y4_init = params["y4_init"][:, -startind:]
        y5_init = params["y5_init"][:, -startind:]

    # Delayed input
    y4_input_d = np.zeros(N)  # delayed input to exc

    np.random.seed(RNGseed)

    # Save the noise in the activity array to save memory
    y4s[:, startind:] = np.random.standard_normal((N, len(t)))
    y5s[:, startind:] = np.random.standard_normal((N, len(t)))

    y0s[:, :startind] = y0_init
    y1s[:, :startind] = y1_init
    y2s[:, :startind] = y2_init
    y3s[:, :startind] = y3_init
    y4s[:, :startind] = y4_init
    y5s[:, :startind] = y5_init
    
    noise_y3 = np.zeros((N,))
    noise_y4 = np.zeros((N,))
    noise_y5 = np.zeros((N,))

    # ------------------------------------------------------------------------
    
    return timeIntegration_njit_elementwise(
        startind,
        t,
        dt,
        sqrt_dt,
        N,
        Cmat,
        K_gl,
        Dmat_ndt,
        y0s,
        y1s,
        y2s,
        y3s,
        y4s,
        y5s,
        ext_input_static,
        ext_input,
        a,
        A,
        b,
        B,
        C,
        C1,
        C2,
        C3,
        C4,
        e0,
        v0,
        r,
        noise_y3,
        noise_y4,
        noise_y5,
        tau_y3,
        tau_y4,
        tau_y5,
        y3_ou,
        y4_ou,
        y5_ou,
        tau_ou,
        sigma_ou,
        y3_ou_mean,
        y4_ou_mean,
        y5_ou_mean,
        y4_input_d,
    )

@numba.njit
def timeIntegration_njit_elementwise(
    startind,
    t,
    dt,
    sqrt_dt,
    N,
    Cmat,
    K_gl,
    Dmat_ndt,
    y0s,
    y1s,
    y2s,
    y3s,
    y4s,
    y5s,
    ext_input_static,
    ext_input,
    a,
    A,
    b,
    B,
    C,
    C1,
    C2,
    C3,
    C4,
    e0,
    v0,
    r,
    noise_y3,
    noise_y4,
    noise_y5,
    tau_y3,    
    tau_y4,
    tau_y5,
    y3_ou,
    y4_ou,
    y5_ou,
    tau_ou,
    sigma_ou,
    y3_ou_mean,
    y4_ou_mean,
    y5_ou_mean,
    y4_input_d,
):
    # Computes the firing rate based on the postsynaptic potential using a sigmoid function
    def Sigm(v):
        x = 2.0 * e0 / (1.0 + np.exp(r * (v0 - v)))
        return x

    for i in range(startind, startind + len(t)):
        # loop through all the nodes
        for no in range(N):
            noise_y3[no] = y3s[no, i]
            noise_y4[no] = y4s[no, i]
            noise_y5[no] = y5s[no, i]

            # delayed input to each node
            y4_input_d[no] = 0

            for l in range(N):
                y4_input_d[no] += K_gl * Cmat[no, l] * (y0s[l, i - Dmat_ndt[no, l] - 1]) # TODO: use y0, y1 or y4 as input?

            # Jansen-Rit model
            y0_rhs = (
                y3s[no, i - 1]
            )
            y1_rhs = (
                y4s[no, i - 1]
            )
            y2_rhs = (
                y5s[no, i - 1]
            )
            y3_rhs = 1/tau_y3 * (
                A * a * Sigm(y1s[no, i - 1] - y2s[no, i - 1])
                - (2.0 * a * y3s[no, i - 1])
                - (a * a * y0s[no, i - 1])
                + y3_ou[no]  # ou noise
            )
            y4_rhs = 1/tau_y4 * (
                A * a * (ext_input_static[no, i - 1] + ext_input[no, i - 1] + y4_input_d[no] + C2 * Sigm(C1 * y0s[no, i - 1]))
                - (2.0 * a * y4s[no, i - 1])
                - (a * a * y1s[no, i - 1])
                + y4_ou[no]  # ou noise
            )
            y5_rhs = 1/tau_y5 * (
                B * b * (C4 * Sigm(C3 * y0s[no, i - 1]))
                - (2.0 * b * y5s[no, i - 1])
                - (b * b * y2s[no, i - 1])
                + y5_ou[no]  # ou noise
            )

            # Euler integration
            y0s[no, i] = y0s[no, i - 1] + dt * y0_rhs
            y1s[no, i] = y1s[no, i - 1] + dt * y1_rhs
            y2s[no, i] = y2s[no, i - 1] + dt * y2_rhs
            y3s[no, i] = y3s[no, i - 1] + dt * y3_rhs
            y4s[no, i] = y4s[no, i - 1] + dt * y4_rhs
            y5s[no, i] = y5s[no, i - 1] + dt * y5_rhs
            
            # Ornstein-Uhlenbeck process
            y3_ou[no] = (
                y3_ou[no] + (y3_ou_mean - y3_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_y3[no]
            ) # mV/ms
            y4_ou[no] = (
                y4_ou[no] + (y4_ou_mean - y4_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_y4[no]
            ) # mV/ms
            y5_ou[no] = (
                y5_ou[no] + (y5_ou_mean - y5_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_y5[no]
            ) # mV/ms

    return t, y0s, y1s, y2s, y3s, y4s, y5s, y3_ou, y4_ou, y5_ou