from . import loadDefaultParams as dp
from . import timeIntegration as ti
from ..model import Model


class JRModel(Model):
    """
    The single-column Jansen-Rit model
    """

    name = "jr"
    description = "Jansen-Rit model"

    init_vars = ['y0_init', 'y1_init', 'y2_init', 'y3_init', 'y4_init', 'y5_init', 'y0_ou', 'y1_ou', 'y2_ou', 'y3_ou', 'y4_ou', 'y5_ou']
    state_vars = ['y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y0_ou', 'y1_ou', 'y2_ou', 'y3_ou', 'y4_ou', 'y5_ou']
    output_vars = ['y0','y1', 'y2']
    default_output = "y0"
    input_vars = ['p_ext']
    default_input = "p_ext"


    def __init__(self, params=None, Cmat=None, Dmat=None, seed=None):
        self.Cmat = Cmat
        self.Dmat = Dmat
        self.seed = seed

        # the integration function must be passed
        integration = ti.timeIntegration

        # load default parameters if none were given
        if params is None:
            params = dp.loadDefaultParams(seed=self.seed)

        # Initialize base class Model
        super().__init__(integration=integration, params=params)
