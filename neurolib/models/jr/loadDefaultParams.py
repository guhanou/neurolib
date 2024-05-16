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

    params.A = 0
    params.a = 0
    params.B = 0
    params.b = 0
    params.C = 0
    params.C1 = 0*params.C
    params.C2 = 0*params.C
    params.C3 = 0*params.C
    params.C4 = 0*params.C
    params.e0 = 0
    params.v = 0
    params.v0 = 0
    params.r = 0

    return params
