import numpy as np

from ...utils.collections import dotdict


def loadDefaultParams(Cmat=None, Dmat=None, seed=None):
    """Load default parameters for the Jansen-Rit model

    """

    params = dotdict({})

    ### runtime parameters
    params.dt = 0.1  # ms 0.1ms is reasonable
    params.duration = 2000  # Simulation duration (ms)
    np.random.seed(seed)  # seed for RNG of noise and ICs
    params.seed = seed

    # external input parameters:
    params.tau_ou = 5.0  # ms Timescale of the Ornstein-Uhlenbeck noise process
    params.sigma_ou = 0.0  # noise intensity
    params.y3_ou_mean = 0.0  # OU process mean
    params.y4_ou_mean = 0.0  # OU process mean
    params.y5_ou_mean = 0.0  # OU process mean

    params.tau_y3 = 10.0  # y3 time constant
    params.tau_y4 = 10.0  # y4 time constant
    params.tau_y5 = 20.0  # y5 time constant

    # model parameters
    params.A = 3.25 # mV
    params.a = 0.1 # kHz
    params.B = 22.0 # mV
    params.b = 0.05 # kHz
    params.C = 135.0 # constant
    params.C1 = 1*params.C
    params.C2 = 0.8*params.C
    params.C3 = 0.25*params.C
    params.C4 = 0.25*params.C
    params.e0 = 0.0025 # kHz
    params.v0 = 6.0 # mV
    params.r = 0.56 # mV

    # signal transmission speed between areas
    params.signalV = 20.0
     # global coupling strength 
    params.K_gl = 1.0 # parameter not yet explored or tested
    
    # connectivity
    if Cmat is None:
        params.N = 1
        params.Cmat = np.zeros((1, 1))
        params.lengthMat = np.zeros((1, 1))
    else:
        params.Cmat = Cmat.copy()  # coupling matrix
        np.fill_diagonal(params.Cmat, 0)  # no self connections
        params.N = len(params.Cmat)  # override number of nodes
        params.lengthMat = Dmat

    # ------------------------------------------------------------------------

    # external input parameters: (pulse density)
    params.ext_input_static = np.full(shape=(params.N, 1), fill_value=0.22) #baseline external input
    params.ext_input = np.random.uniform(-0.1, 0.1, (params.N, 1)) #random variation in external input
    
    params.y0_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))
    params.y1_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))
    params.y2_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))
    params.y3_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))
    params.y4_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))
    params.y5_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))
    
    # Ornstein-Uhlenbeck noise state variables
    params.y3_ou = np.zeros((params.N,))
    params.y4_ou = np.zeros((params.N,))
    params.y5_ou = np.zeros((params.N,))
  
    return params
