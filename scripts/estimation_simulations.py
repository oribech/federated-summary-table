import multiprocessing
import os
import time
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property, partial
from pathlib import Path
from typing import Tuple, List

import numba as nb
import numpy as np
import pandas as pd
from numba import njit
from scipy import optimize
from scipy import stats
from tqdm import tqdm

from scripts import utils
from scripts.config import Configurator
from scripts.federated_binning import TabularSummary
from scripts.utils import make_str_ok_for_file


def quantile_loss(u, arr, tau):
    msk = arr > u
    return tau * sum(arr[msk] - u) + (tau - 1) * sum(arr[~msk] - u)


def federated_quantile_loss(u, data, tau):
    return sum(
        quantile_loss(u, data[i][0], tau)
        for i in range(len(data))

    )


@njit
def federated_loglik_helper(lmbda, data, sd, n, ncenters):
    loglik = sum([
        sum(
            (lmbda - 1) * (np.sign(data[center_i]) * np.log(np.abs(data[center_i]) + 1))
        )
        for center_i in range(ncenters)
    ])
    return loglik - n / 2 * np.log(sd ** 2)


def federated_moments_helper(lmbda, data, ncenters, n):
    sum_all = sum([
        sum(utils.yeo_johnson(data[center_i], lmbda))
        for center_i in range(ncenters)
    ])
    sum_sq_all = sum([
        sum(utils.yeo_johnson(data[center_i], lmbda) ** 2)
        for center_i in range(ncenters)
    ])

    mu, sd = sum_all / n, (sum_sq_all / n - (sum_all / n) ** 2) ** .5

    return mu, sd


@dataclass
class EstimationSimulation:
    """
        This class is used to simulate data for the estimation simulations.

        Parameters
        ----------
        number_of_centers: int
            The number of centers to be considered in the simulation.
        nx: int
              Number of data points in the control.
        ny: int
         	Number of data points in the treatment.
        shape: float
            The shape parameter for the gamma distribution used in the simulation.
        mu1: float
            The mean of the normal distribution used in the simulation for the control group.
        mu2: float
            The mean of the normal distribution used in the simulation for the treatment group.
        sd1: float
        sd2: float
        privacy_condition: int
            A parameter indicating the privacy condition to be used in the simulation.
        quantiles: list
            A list of quantiles to be considered in the simulation.
        model: A statistical model from the scipy.stats module, default is stats.gamma
            The statistical model to be used in the simulation.

        Methods
        -------
        Numerous methods are defined for various parts of the simulation, including generating data, calculating quantiles,
        estimating parameters, and more. Each method has its own specific purpose and arguments, and they are used together
        to perform the complete simulation.
        """
    number_of_centers: int
    nx: int
    ny: int
    shape: float
    mu1: float
    mu2: float
    sd1: float
    sd2: float
    privacy_condition: int
    quantiles: list
    model = stats.gamma

    def map_shifts_to_gamma(self, q, mu, sd):
        xq = self.model.ppf(q=q, a=self.shape, scale=1)
        mut = np.log((xq + mu) / xq)
        sdt = (1 / 2) * np.log((xq + mu + sd) / (xq + mu - sd))
        return mut, sdt

    @cached_property
    def center_obs(self) -> Tuple:
        return utils.get_allocation(n=self.nx, k=self.number_of_centers), []

    @cached_property
    def center_shifts(self):
        shift_lst1 = []
        for _ in range(self.number_of_centers):
            eta1, sd1 = self.map_shifts_to_gamma(q=0.5, mu=self.mu1, sd=self.sd1)
            mu1 = stats.norm.rvs(size=1, loc=eta1, scale=sd1)
            shift_lst1.append(np.exp(mu1).item())
        return shift_lst1, []

    @cached_property
    def data(self) -> List[Tuple]:
        data_lst = []
        for nx, eta1 in zip(self.center_obs[0], self.center_shifts[0]):
            assert self.shape is not None, "Shape got to be float>0"
            x1 = self.model.rvs(size=nx, a=self.shape, scale=eta1)
            x2 = []
            data_lst.append((x1, x2))
        return data_lst

    @cached_property
    def data_flat(self):
        return [np.array(self.data[i][0]) for i in range(self.number_of_centers)]

    @cached_property
    def x1(self):
        x = []
        for c in self.data:
            x.extend(c[0])
        return x

    @cached_property
    def tabular_summary_helper(self):
        return TabularSummary(privacy_condition=self.privacy_condition, centers_data=self.data).tabular_summary

    @cached_property
    def tabular_summary(self):
        bins, f1, f2, _, _ = self.tabular_summary_helper
        return bins, f1, f2

    @cached_property
    def tabular_summary_tail_bins_included(self):
        bins, f1, f2, last_bin_max, first_bin_min = self.tabular_summary_helper
        bins = [first_bin_min] + bins
        bins[-1] = last_bin_max
        f1, f2 = [0] + f1, [0] + f2
        return bins, f1, f2

    def real_quantile(self, q):
        return utils.mixed_gamma_quantile(
            shapes=np.repeat(self.shape, self.number_of_centers),
            scales=self.center_shifts[0],
            ns=self.center_obs[0],
            q=q,
        )

    def federated_moments(self, lmbda):
        data = self.data_flat
        n = self.nx
        ncenters = self.number_of_centers
        mu, sd = federated_moments_helper(lmbda, data, ncenters, n)
        return mu, sd

    def federated_loglik(self, lmbda):
        mu, sd = self.federated_moments(lmbda)
        return federated_loglik_helper(lmbda, nb.typed.List(self.data_flat), sd, self.nx, self.number_of_centers)

    @cached_property
    def federated_yj_lmbda(self):
        brack = (-2, 2)
        neg_log_lik = lambda lmbda: -1 * self.federated_loglik(lmbda)
        return optimize.brent(neg_log_lik, brack=brack)

    def normal_quantile(self, q):
        lm = self.federated_yj_lmbda
        mu, sd = self.federated_moments(lm)
        normal_q = mu + sd * stats.norm.ppf(q)
        return utils.inv_yeo_johnson([normal_q], lm).item()

    def reg_quantile(self, q) -> float:
        assert 0 < q < 1, "q has to be in (0,1)"

        def loss(u, data, tau):
            return federated_quantile_loss(u, data, tau)

        _, _, _, last_bin_max, last_bin_min = self.tabular_summary_helper
        brack = (last_bin_min, last_bin_max)
        return optimize.brent(loss, brack=brack, args=(self.data, q))

    def agg_quantile(self, q):
        n = 0
        q_i = 0
        for center_i in range(self.number_of_centers):
            center_data = self.data[center_i][0]
            w = len(center_data)
            q_i += np.quantile(self.x1, q) * w
            n += w
        return q_i / n

    def sample_quantile(self, q):
        return np.quantile(self.x1, q)

    def sample_mle_quantile(self, q):
        dist = self.model
        params1 = dist.fit(self.x1)
        return dist.ppf(q, *params1)

    def yj_quantile(self, q):
        return utils.yj_quantile(
            freq=self.tabular_summary[1], bins=self.tabular_summary[0], q=q
        ).item()

    def interpolation_extrapolation_quantile(self, q):
        return utils.interpolation_quantile(
            freq=self.tabular_summary[1], bins=self.tabular_summary[0], q=q, include_last_bin=False,
        )

    def interpolation_quantile(self, q):
        return utils.interpolation_quantile(
            freq=self.tabular_summary_tail_bins_included[1], bins=self.tabular_summary_tail_bins_included[0], q=q,
            include_last_bin=True,
        )

    def quantile_errs(self, q):
        real_quantile1 = self.real_quantile(q)
        return dict(
            shape=self.shape,
            priors=f"{(self.mu1, self.sd1, self.mu2, self.sd2)}",
            ncenters=self.number_of_centers,
            p=q,
            qi=real_quantile1,
            q=utils.mixed_gamma_quantile([self.shape], [1], [1], q=q),
            sample_quantile=self.sample_quantile(q) - real_quantile1,
            yj_quantile=self.yj_quantile(q) - real_quantile1,
            interpolation_quantile=self.interpolation_quantile(q) - real_quantile1,
            interpolation_extrapolation_quantile=self.interpolation_extrapolation_quantile(q) - real_quantile1,
            reg_quantile=self.reg_quantile(q) - real_quantile1,
            normal_quantile=self.normal_quantile(q) - real_quantile1,
        )

    @cached_property
    def errs(self) -> pd.DataFrame:
        df = []
        for q in self.quantiles:
            df.append(self.quantile_errs(q=q))
        df = pd.DataFrame.from_records(df)
        return df

    @cached_property
    def quantile_table(self):
        methods = ['agg_quantile', 'interpolation_extrapolation_quantile', 'interpolation_quantile', 'normal_quantile',
                   'real_quantile', 'reg_quantile', 'sample_mle_quantile', 'sample_quantile', 'yj_quantile']

        return pd.DataFrame({
            method: [getattr(self, method)(q) for q in self.quantiles]
            for method in methods
        },
            index=pd.Index(self.quantiles),

        )


def _simulation(i, shape, quantiles, number_of_centers, mu1, mu2, sd1, sd2, nx, ny, privacy_condition):
    return EstimationSimulation(
        nx=nx,
        ny=ny,
        number_of_centers=number_of_centers,
        shape=shape,
        mu1=mu1,
        mu2=mu2,
        sd1=sd1,
        sd2=sd2,
        privacy_condition=privacy_condition,
        quantiles=quantiles,
    ).errs


@dataclass
class EstimationSimulator:
    """
    The EstimationSimulator class is responsible for handling the execution of estimation simulations.

    This class uses multiprocessing to perform simulations under different conditions defined by the configuration.

    Attributes
    ----------
    config : Configurator
        An instance of the Configurator class that holds the configuration parameters for the simulations.

    Methods
    -------
    run_simulation():
        Executes the simulation based on the provided configuration parameters.
    """
    config: Configurator
    mod: int = 100

    @property
    def iterations_number(self):
        return len(self.config.ncenters) * len(self.config.shapes) * len(
            self.config.priors) * self.config.n_simulations * len(self.config.quantiles)

    @cached_property
    def simulations_data(self):
        df = pd.DataFrame()
        i = 0
        time0 = time.time()
        for ncenter in self.config.ncenters:
            for shape in self.config.shapes:
                for priors in self.config.priors:
                    for _ in range(self.config.n_simulations):
                        mu1, sd1, mu2, sd2 = priors
                        simulation = EstimationSimulation(
                            nx=self.config.nx,
                            ny=self.config.ny,
                            number_of_centers=ncenter,
                            shape=shape,
                            mu1=mu1,
                            mu2=mu2,
                            sd1=sd1,
                            sd2=sd2,
                            privacy_condition=self.config.privacy_condition,
                            quantiles=self.config.quantiles,
                        )

                        if i % self.mod == 0:
                            time1 = time.time() - time0
                            avg_time = time1 / (i + 1)
                            print(
                                f"Simulation {i} out of {self.iterations_number}, running for {time1:.4f}",
                                f"ETA {avg_time * self.iterations_number - time1:.4f} secs"

                            )

                        errs = simulation.errs
                        errs["simulation_number"] = i
                        df = df.append(errs, ignore_index=True)
                        i += 1
        return df

    def simulations_data_parallel(self):
        df = []
        i = 0
        time0 = time.time()
        for ncenter in self.config.ncenters:
            for shape in self.config.shapes:
                for priors in self.config.priors:
                    mu1, sd1, mu2, sd2 = priors
                    nx = self.config.nx
                    ny = self.config.ny
                    privacy_condition = self.config.privacy_condition
                    simulation = partial(
                        _simulation,
                        shape=shape,
                        quantiles=self.config.quantiles,
                        number_of_centers=ncenter,
                        mu1=mu1,
                        mu2=mu2,
                        sd1=sd1,
                        sd2=sd2,
                        nx=nx,
                        ny=ny,
                        privacy_condition=privacy_condition
                    )
                    processes = os.cpu_count() - 2

                    lst = list(range(self.config.n_simulations))
                    with multiprocessing.Pool(processes) as pool:
                        errs = pd.concat(list(tqdm(pool.imap(simulation, lst), total=len(lst), )))

                    errs['simulation_number'] = [j + i for j in range(len(errs))]
                    i += len(errs)
                    errs['priors'] = f"{(mu1, sd1, mu2, sd2)}"
                    df.append(errs)

                    if i % self.mod == 0:
                        time1 = time.time() - time0
                        avg_time = time1 / (i + 1)

                        print(
                            f"Simulation {i} out of {self.iterations_number}, running for {time1:.4f}",
                            f"ETA {avg_time * self.iterations_number - time1:.4f} secs"

                        )
        df = pd.concat(df, ignore_index=True)
        return df

    @property
    def results(self):
        df = self.simulations_data_parallel()
        molten_df = df.melt(
            id_vars=['q', 'shape', 'ncenters', 'simulation_number', 'priors', 'p', 'qi'],
            value_vars=[col for col in df.columns if col.endswith('_quantile')],
            var_name='Method',
        )
        return molten_df

    @cached_property
    def root_path(self):
        return Path.cwd().parent

    @property
    def data_path(self):
        return self.root_path / 'data' / 'estimation'

    @cached_property
    def results_path(self):
        from datetime import datetime
        pth = self.data_path / make_str_ok_for_file("{date}".format(date=datetime.today()))
        Path.mkdir(pth, parents=True, exist_ok=True)
        return pth

    def run_simulations(self):

        print(f"Initiating: {datetime.now()=}")
        time0 = time.time()
        # simulator.errs_boxplots()
        csv_path = self.results_path / 'simulations.csv'

        df = self.results

        df.to_csv(csv_path, index=False)
        time1 = time.time()
        print(f"Finish in: {time1 - time0} secs")
        print(f'File saved in: {csv_path}')
        print(f"{datetime.now()=}")


def run_main():
    np.random.seed(123)
    EstimationSimulator(
        config=Configurator(
            privacy_condition=10,
            n_simulations=2000,
            nx=1500,
            ny=1500,
            quantiles=[0.02, 0.25, 0.5, 0.75, 0.98],
            ncenters=[3, 5, 10],
            shapes=[4, 10],
            priors=[(0, 0.1, 0, 0)]

        )).run_simulations()


if __name__ == '__main__':
    run_main()
