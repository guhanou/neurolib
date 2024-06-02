from . import loadDefaultParams as dp
from . import timeIntegration as ti
from ..model import Model


class JRModel(Model):
    """
    The single-column Jansen-Rit model
    """

    name = "jr"
    description = "Jansen-Rit model"

    init_vars = ['y0_init', 'y1_init', 'y2_init', 'y3_init', 'y4_init', 'y5_init']
    state_vars = ['y0', 'y1', 'y2', 'y3', 'y4', 'y5']
    output_vars = ['y0', 'y2']
    default_output = "y0"
    input_vars = ['p_ext']
    default_input = "p_ext"

    # because this is not a rate model, the input

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
