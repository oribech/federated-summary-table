import multiprocessing
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property, partial
from pathlib import Path
from typing import Sequence, Tuple, Any

import numpy as np
import pandas as pd
import scipy as sc
from scipy import stats
from scipy.stats import chi2
from scipy.stats import rankdata
from tqdm import tqdm

from scripts import utils
from scripts.config import Configurator
from scripts.config import MetaData
from scripts.federated_binning import TabularSummary
from scripts.utils import make_str_ok_for_file


def mwu_sd(nx1, nx2, ties=None):
    if ties:
        n = nx1 + nx2
        se1 = (nx1 * nx2 / 12)
        se2 = sum(ties ** 3 - ties) / (n * (n - 1))
        se = 2 * (se1 * ((n + 1) - se2)) ** .5
        return se
    return (nx1 * nx2 * (nx1 + nx2 + 1) / 12) ** .5


def score(i, j):
    if i > j:
        return 1
    elif i == j:
        return 0
    return -1


def tabular_statistic(tabular_summary):
    running_sum = 0
    for i in range(len(tabular_summary[0])):
        for j in range(len(tabular_summary[0])):
            fxi = tabular_summary[1][i]
            fyj = tabular_summary[2][j]
            running_sum += fxi * fyj * score(i, j)
    return running_sum


def map_params_to_gamma(mu, sigma):
    alpha, beta = (mu / sigma) ** 2, (mu / sigma ** 2)
    return alpha, 1 / beta


@dataclass
class Mwu:
    x1: Sequence
    x2: Sequence
    correct_continuity: bool = False

    def __post_init__(self):
        self.x1 = np.array(self.x1)
        self.x2 = np.array(self.x2)

    @cached_property
    def n1(self):
        return len(self.x1)

    @cached_property
    def n2(self):
        return len(self.x2)

    @cached_property
    def mu(self):
        return self.n1 * self.n2 / 2

    @cached_property
    def sd(self):
        """
        SD for transformed MWU stat 2(MWU-n1*n2/2)
        """
        return 2 * mwu_sd(self.n1, self.n2)

    @cached_property
    def stat(self):
        n1, n2 = self.n1, self.n2
        xy = np.concatenate((self.x1, self.x2))
        ranks = rankdata(xy, axis=-1)  # method 2, step 1
        R1 = ranks[..., :n1].sum(axis=-1)  # method 2, step 2
        U1 = R1 - n1 * (n1 + 1) / 2
        if self.correct_continuity:
            return 2 * (U1 - self.mu + .5)
        return 2 * (U1 - self.mu)

    @cached_property
    def z(self):
        return self.stat / self.sd

    @cached_property
    def pval(self):
        """
        P-val for x1<x2
        """
        z = self.z
        # return 2 * (min(1 - sc.stats.norm.cdf(z), sc.stats.norm.cdf(z)))
        return sc.stats.norm.cdf(z)

    @cached_property
    def weight(self):
        return self.n1 * self.n2 / self.sd


@dataclass
class TestingSimulation:
    """
        This class is used to simulate data for the testing simulations.

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

        """
    number_of_centers: int
    nx: int
    ny: int
    mu1: float
    mu2: float
    sd1: float
    sd2: float
    privacy_condition: int

    @cached_property
    def center_obs(self) -> Tuple[Any, Any]:
        return (
            utils.get_allocation(n=self.nx, k=self.number_of_centers),
            utils.get_allocation(n=self.ny, k=self.number_of_centers)
        )

    @cached_property
    def center_shifts(self):
        shift_lst1 = []
        shift_lst2 = []
        for _ in range(self.number_of_centers):
            mu1 = stats.norm.rvs(size=1, loc=self.mu1, scale=self.sd1)
            mu2 = stats.norm.rvs(size=1, loc=self.mu2, scale=self.sd2)
            shift_lst1.append(mu1.item())
            shift_lst2.append(mu2.item())
        return shift_lst1, shift_lst2

    @cached_property
    def data(self):
        data_lst = []
        for nx, ny, eta1, eta2 in zip(*self.center_obs, *self.center_shifts):
            x1 = stats.norm.rvs(size=nx) + eta1
            x2 = stats.norm.rvs(size=ny) + eta2 + eta1
            data_lst.append((x1, x2))
        return data_lst

    @cached_property
    def centers_data(self):
        return self.data

    @cached_property
    def tabular_summary_helper(self):
        return TabularSummary(privacy_condition=self.privacy_condition, centers_data=self.data).tabular_summary

    @cached_property
    def tabular_summary(self):
        bins, f1, f2, _, _ = self.tabular_summary_helper
        return bins, f1, f2

    @cached_property
    def tabular_summary_last_bin_included(self):
        bins, f1, f2, last_bin_max, first_bin_min = self.tabular_summary_helper
        bins[-1] = last_bin_max
        return bins, f1, f2

    @cached_property
    def x1(self):
        x = []
        for c in self.centers_data:
            x.extend(c[0])
        return x

    @cached_property
    def x2(self):
        x = []
        for c in self.centers_data:
            x.extend(c[1])
        return x

    @cached_property
    def unconstrained_pval(self):
        return Mwu(self.x1, self.x2).pval

    @cached_property
    def center_stats(self):
        f"""helps to see the different effects between centers"""
        stats = []
        for center_i in range(self.number_of_centers):
            x1, x2 = self.centers_data[center_i]
            stats.append(Mwu(x1, x2))
        return stats

    @cached_property
    def fisher_pval(self):
        p_vals = [m.pval for m in self.center_stats]
        return 1 - chi2.cdf(-2 * sum(np.log(p_vals)), df=2 * len(p_vals))

    @cached_property
    def sum_pval(self):
        """
        P-val for x1<x2
        """
        statistic = 0
        sum_var = 0
        for center_i in range(self.number_of_centers):
            mwu = self.center_stats[center_i]
            statistic += mwu.stat
            sum_var += mwu.sd ** 2
        z = statistic / sum_var ** .5
        # return 2 * (min(1 - sc.stats.norm.cdf(z), sc.stats.norm.cdf(z)))
        return sc.stats.norm.cdf(z)

    @cached_property
    def weighted_pval(self):
        """
        P-val for x1<x2
        """
        sum_wsq = 0
        sum_z = 0
        for center_i in range(self.number_of_centers):
            mwu = self.center_stats[center_i]
            w = mwu.weight
            sum_wsq += w ** 2
            sum_z += w * mwu.z
        z = sum_z / sum_wsq ** .5
        # return 2 * (min(1 - sc.stats.norm.cdf(z), sc.stats.norm.cdf(z)))
        return sc.stats.norm.cdf(z)

    @cached_property
    def tabular_pval(self):
        fx1 = self.tabular_summary[1]
        fx2 = self.tabular_summary[2]
        statistics = tabular_statistic(self.tabular_summary)
        freqs = np.array(fx1) + np.array(fx2)
        nx1, nx2 = sum(fx1), sum(fx2)
        n = nx1 + nx2
        se1 = (nx1 * nx2 / 12)
        se2 = sum(freqs ** 3 - freqs) / (n * (n - 1))
        se = 2 * (se1 * ((n + 1) - se2)) ** .5
        z = statistics / se
        # return 2 * (min(1 - sc.stats.norm.cdf(z), sc.stats.norm.cdf(z)))
        return sc.stats.norm.cdf(z)

    @cached_property
    def pvals(self) -> dict:
        return dict(
            ncenters=self.number_of_centers,
            unconstrained_pval=self.unconstrained_pval,
            fisher_pval=self.fisher_pval,
            sum_pval=self.sum_pval,
            weighted_pval=self.weighted_pval,
            tabular_pval=self.tabular_pval,
        )


def _simulation(i, number_of_centers, mu1, mu2, sd1, sd2, nx, ny, privacy_condition):
    return TestingSimulation(
        number_of_centers=number_of_centers,
        nx=nx,
        ny=ny,
        mu1=mu1,
        mu2=mu2,
        sd1=sd1,
        sd2=sd2,
        privacy_condition=privacy_condition
    ).pvals


class TestingSimulations:
    """
    The TestingSimulations class is responsible for handling the execution of testing simulations.

    """

    def __init__(self, config: Configurator, mod: int = 100, n_jobs: int = 5):
        self.config = config
        self.mod: int = mod
        self.n_jobs: int = n_jobs

    @cached_property
    def plots_path(self):
        return Path.cwd().parent / 'plots'

    @property
    def iterations_number(self):
        return len(self.config.ncenters) * len(self.config.priors) * self.config.n_simulations

    @cached_property
    def pvals_df(self):
        df = pd.DataFrame()
        i = 0
        time0 = time.time()
        for ncenter in self.config.ncenters:
            for priors in self.config.priors:
                for _ in range(self.config.n_simulations):
                    i += 1
                    mu1, sd1, mu2, sd2 = priors
                    simulation = TestingSimulation(
                        number_of_centers=ncenter,
                        nx=self.config.nx,
                        ny=self.config.ny,
                        mu1=mu1,
                        mu2=mu2,
                        sd1=sd1,
                        sd2=sd2,
                        privacy_condition=self.config.privacy_condition
                    )
                    if i % self.mod == 0:
                        time1 = time.time() - time0
                        avg_time = time1 / (i + 1)
                        print(
                            f"Simulation {i} out of {self.iterations_number}, running for {time1:.4f}",
                            f"ETA {avg_time * self.iterations_number - time1:.4f} secs"

                        )
                    pvals = simulation.pvals
                    pvals['simulation_number'] = i
                    dfi = pd.DataFrame(pvals, index=[i])
                    dfi['priors'] = f"{(mu1, sd1, mu2, sd2)}"
                    df = df.append(dfi, ignore_index=True)
        return df

    def pvals_df_parallel(self):
        df = []
        i = 0
        time0 = time.time()
        for ncenter in self.config.ncenters:
            for priors in self.config.priors:
                mu1, sd1, mu2, sd2 = priors
                nx = self.config.nx
                ny = self.config.ny
                privacy_condition = self.config.privacy_condition
                simulation = partial(_simulation, number_of_centers=ncenter, mu1=mu1, mu2=mu2, sd1=sd1, sd2=sd2, nx=nx,
                                     ny=ny, privacy_condition=privacy_condition)
                processes = os.cpu_count() - 2

                lst = list(range(self.config.n_simulations))
                with multiprocessing.Pool(processes) as pool:
                    pvals = list(tqdm(pool.imap(simulation, lst), total=len(lst), ))
                pvals = pd.DataFrame(pvals)
                pvals['simulation_number'] = [j + i for j in range(len(pvals))]
                i += len(pvals)
                pvals['priors'] = f"{(mu1, sd1, mu2, sd2)}"
                df.append(pvals)

                if i % self.mod == 0:
                    time1 = time.time() - time0
                    avg_time = time1 / (i + 1)

                    print(
                        f"Simulation {i} out of {self.iterations_number}, running for {time1:.4f}",
                        f"ETA {avg_time * self.iterations_number - time1:.4f} secs"

                    )
        df = pd.concat(df)
        return df

    def save(self, fpath):
        # create a pickle file
        picklefile = open(fpath, 'wb')
        # pickle the dictionary and write it to file
        pickle.dump(self, picklefile)
        # close the file
        picklefile.close()


@dataclass
class PickleHandler:

    def load(self, opath):
        picklefile = open(opath, 'rb')
        # unpickle the dataframe
        obj = pickle.load(picklefile)
        # close file
        picklefile.close()
        return obj


class TestingRunner:
    """
    Runs the testing simulations with different configurations and saves the results
    """

    @cached_property
    def root_path(self):
        return Path.cwd().parent

    @property
    def data_path(self):
        return (self.root_path / 'data').as_posix()

    @cached_property
    def results_path(self):
        from datetime import datetime
        pth = Path(self.data_path) / make_str_ok_for_file("{date}".format(date=datetime.today()))
        Path.mkdir(pth, parents=True, exist_ok=True)
        return pth

    def power_comparison(self):
        config = Configurator(
            privacy_condition=10,
            n_simulations=2000,
            nx=1500,
            ny=1500,
            ncenters=[3, 5, 10],
            priors=MetaData(mode='power_comparison').testing_hyper_params
        )

        testing_simulator = TestingSimulations(
            config=config,
        )

        time0 = time.time()
        pickel_path = (self.results_path / 'testing_simulations_power').as_posix()
        csv_path = self.results_path / 'testing_simulations_power.csv'
        testing_simulator.pvals_df_parallel().to_csv(csv_path, index=False)
        testing_simulator.save(pickel_path)
        time1 = time.time()
        print(f"{time1 - time0}")

    def run_method_comparison(self):
        np.random.seed(1)

        config = Configurator(
            privacy_condition=10,
            n_simulations=2000,
            nx=1500,
            ny=1500,
            ncenters=[3, 5, 10],
            priors=MetaData(mode='method_comparison').testing_hyper_params
        )

        testing_simulator = TestingSimulations(
            config=config,
        )

        time0 = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        fname = f'testing_simulations_{timestamp}'
        pickel_path = (self.results_path / fname).as_posix()
        csv_path = self.results_path / f'{fname}.csv'
        testing_simulator.pvals_df_parallel().to_csv(csv_path, index=False)
        testing_simulator.save(pickel_path)
        time1 = time.time()
        print(f"{time1 - time0}")


def run_main():
    TestingRunner().run_method_comparison()


if __name__ == '__main__':
    run_main()
