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
    tau_exc = params["tau_exc"]  
    tau_inh = params["tau_inh"]  

    # external input parameters:
    # Parameter of the Ornstein-Uhlenbeck process for the external input(ms)
    tau_ou = params["tau_ou"]
    # Parameter of the Ornstein-Uhlenbeck (OU) process for the external input ( mV/ms/sqrt(ms) )
    sigma_ou = params["sigma_ou"]
    # Mean external excitatory input (OU process)
    exc_ou_mean = params["exc_ou_mean"]
    # Mean external inhibitory input (OU process)
    inh_ou_mean = params["inh_ou_mean"]

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
    exc_ou = params["exc_ou"].copy()
    inh_ou = params["inh_ou"].copy()

    # state variable arrays, have length of t + startind
    # they store initial conditions AND simulated data
    y0s = np.zeros((N, startind + len(t)))
    y1s = np.zeros((N, startind + len(t)))
    y2s = np.zeros((N, startind + len(t)))
    y3s = np.zeros((N, startind + len(t)))
    y4s = np.zeros((N, startind + len(t)))
    y5s = np.zeros((N, startind + len(t)))

    # External input param
    p_ext_static = mu.adjustArrayShape(params["p_ext_static"], y4s)
    p_ext_variation = mu.adjustArrayShape(params["p_ext_variation"], y4s)

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

    # TODO: Delayed input activity

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
    
    noise_exc = np.zeros((N,))
    noise_inh = np.zeros((N,))

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
        p_ext_static,
        p_ext_variation,
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
        noise_exc,
        noise_inh,
        tau_exc,
        tau_inh,
        exc_ou,
        inh_ou,
        tau_ou,
        sigma_ou,
        exc_ou_mean,
        inh_ou_mean,
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
    p_ext_static,
    p_ext_variation,
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
    noise_exc,
    noise_inh,
    tau_exc,
    tau_inh,
    exc_ou,
    inh_ou,
    tau_ou,
    sigma_ou,
    exc_ou_mean,
    inh_ou_mean,
):
    
    def Sigm(v):
        x = 2.0 * e0 / (1.0 + np.exp(r * (v0 - v)))
        return x
    
    for i in range(startind, startind + len(t)):
        # loop through all the nodes
        for no in range(N):
            noise_exc[no] = y4s[no, i]
            noise_inh[no] = y5s[no, i]

            # TODO: delayed input -> not applicable for Jansen-Rit model?

            # Jansen-Rit model
            # Implmentation without consideration of possible tau-values
            y0_rhs = (
                y3s[no, i - 1]
            )
            y1_rhs = (
                y4s[no, i - 1]
            )
            y2_rhs = (
                y5s[no, i - 1]
            )
            y3_rhs = (
                A * a * Sigm(y1s[no, i - 1] - y2s[no, i - 1])
                - (2.0 * a * y3s[no, i - 1])
                - (a * a * y0s[no, i - 1])
            )
            
            y4_rhs = 1/tau_exc * (
                A * a * (p_ext_static[no, i - 1] + p_ext_variation[no, i - 1] + C2 * Sigm(C1 * y0s[no, i - 1]))
                - (2.0 * a * y4s[no, i - 1])
                - (a * a * y1s[no, i - 1])
                + exc_ou[no]  # ou noise
            )
            y5_rhs = 1/tau_inh * (
                B * b * (C4 * Sigm(C3 * y0s[no, i - 1]))
                - (2.0 * b * y5s[no, i - 1])
                - (b * b * y2s[no, i - 1])
                + inh_ou[no]  # ou noise
            )

            # Euler integration
            y0s[no, i] = y0s[no, i - 1] + dt * y0_rhs
            y1s[no, i] = y1s[no, i - 1] + dt * y1_rhs
            y2s[no, i] = y2s[no, i - 1] + dt * y2_rhs
            y3s[no, i] = y3s[no, i - 1] + dt * y3_rhs
            y4s[no, i] = y4s[no, i - 1] + dt * y4_rhs
            y5s[no, i] = y5s[no, i - 1] + dt * y5_rhs

            """
            # make sure state variables do not exceed 1 (can only happen with noise)
            def preventExceed(x): 
                if x > 1.0:
                    x = 1.0
                elif x < 0.0:
                    x = 0.0
                return x

            y0s[no, i] = preventExceed(y0s[no, i])
            y1s[no, i] = preventExceed(y1s[no, i])
            y2s[no, i] = preventExceed(y2s[no, i])
            y3s[no, i] = preventExceed(y3s[no, i])
            y4s[no, i] = preventExceed(y4s[no, i])
            y5s[no, i] = preventExceed(y5s[no, i])
            """

            # Ornstein-Uhlenbeck process
            exc_ou[no] = (
                exc_ou[no] + (exc_ou_mean - exc_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_exc[no]
            )  # mV/ms
            inh_ou[no] = (
                inh_ou[no] + (inh_ou_mean - inh_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_inh[no]
            )  # mV/ms

    return t, y0s, y1s, y2s, exc_ou, inh_ou