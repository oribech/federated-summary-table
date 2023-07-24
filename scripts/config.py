from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from scipy import stats


@dataclass
class Configurator:
    n_simulations: int  # = 10
    nx: int  # = 1500
    ny: int  # = 1500

    quantiles: list = field(default_factory=lambda: [0.02, 0.25, 0.5, 0.75, 0.98])
    ncenters: list = field(default_factory=lambda: [3, 5, 10])
    shapes: list = field(default_factory=lambda: [4, 10])
    priors: list = field(default_factory=lambda: [(0, 0.1, 0, 0)])  # [(mu1,sd1,mu2,sd2)]

    privacy_condition: int = 10
    model: stats.rv_continuous = stats.norm
    testing: bool = False

    @cached_property
    def quantile_errs_by_ncenters_shapes_path(self):
        return Path.cwd().parent / 'data' / 'quantile_errs_by_n_centers_shapes_path.csv'


def compact_to_complete(compact_form):
    complete_form = []
    for param_set in compact_form:
        param_dict = dict(zip(['mu2', 'sd1', 'sd2'], list(param_set)))
        complete_form.append((0, param_dict['sd1'], param_dict['mu2'], param_dict['sd2']))
    return complete_form


class MetaData:

    def __init__(self, mode: str = 'method_comparison'):

        """
        mode: {'method_comparison','power_comparison'}
        """

        self.mode = mode

    @property
    def testing_hyper_params_compact(self):
        if self.mode == 'method_comparison':
            return [
                (0, .1, 0),
                (0, .1, 0.05),
                (.05, 0, 0),
                (.05, 0, .05),
                (.05, .1, 0),
                (.05, .2, 0),
                (.05, .1, .05),
                (.05, .1, .1)
            ]

        elif self.mode == 'power_comparison':
            return [

                (0.05, 0.1, 0.05),
                (0.0625, 0.1, 0.05),
                (0.075, 0.1, 0.05),
                (0.0875, 0.1, 0.05),
                (0.1, 0.1, 0.05),
            ]
        assert False, 'mode not supported'

    @property
    def testing_hyper_params(self):
        raw_form = self.testing_hyper_params_compact
        return compact_to_complete(raw_form)
