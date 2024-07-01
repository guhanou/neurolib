from . import loadDefaultParams as dp
from . import timeIntegration as ti
from ..model import Model


class JRModel(Model):
    """
    The single-column Jansen-Rit model

    Based on: Jansen, B.H., Rit, V.G. Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns. Biol. Cybern. 73, 357–366 (1995).
    """
    name = "jr"
    description = "Jansen-Rit model"

    init_vars = ['y0_init', 'y1_init', 'y2_init', 'y3_init', 'y4_init', 'y5_init', "y3_ou", "y4_ou", "y5_ou"]
    state_vars = ['y0', 'y1', 'y2', 'y3', 'y4', 'y5', "y3_ou", "y4_ou", "y5_ou"]
    output_vars = ['y0','y1', 'y2']
    default_output = "y0"
    input_vars = ['ext_input']
    default_input = "ext_input"

    def __init__(self, params=None, Cmat=None, Dmat=None, seed=None):
        self.Cmat = Cmat
        self.Dmat = Dmat
        self.seed = seed

        # the integration function must be passed
        integration = ti.timeIntegration

        # load default parameters if none were given
        if params is None:
            params = dp.loadDefaultParams(Cmat=self.Cmat, Dmat=self.Dmat, seed=self.seed)

        # Initialize base class Model
        super().__init__(integration=integration, params=params)
