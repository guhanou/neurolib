from . import loadDefaultParams as dp
from . import timeIntegration as ti
from ..model import Model


class WCModel(Model):
    """
    The single-column Jansen-Rit model
    """

    name = "jr"
    description = "Jansen-Rit model"

    init_vars = []
    state_vars = []
    output_vars = []
    default_output = ""
    input_vars = []
    default_input = ""

    # because this is not a rate model, the input

    def __init__(self, params=None, seed=None):

        self.seed = seed

        # the integration function must be passed
        integration = ti.timeIntegration

        # load default parameters if none were given
        if params is None:
            params = dp.loadDefaultParams(seed=self.seed)

        # Initialize base class Model
        super().__init__(integration=integration, params=params)
