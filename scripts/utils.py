import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy
from scipy import interpolate
from scipy import stats
from scipy.optimize import fsolve
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from scipy.stats import yeojohnson


def tabular_ecdf(tabular_summary):
    datum = pd.DataFrame(tabular_summary).transpose()
    return datum[0], datum[1].cumsum() / datum[1].sum(), datum[2].cumsum() / datum[2].sum()


def interpolation_quantile(freq, bins, q, include_last_bin, **kwargs):
    """
    """
    if include_last_bin:
        freqs = pd.Series(freq)
        bins = pd.Series(bins)
    else:
        freqs = pd.Series(freq[:-1])
        bins = pd.Series(bins[:-1])
    rfreq = freqs.cumsum() / sum(freq)
    # excluding problematic bins
    f = interpolate.interp1d(rfreq, bins, fill_value='extrapolate', **kwargs)
    # transformation

    return f(q).item()


def inv_yeo_johnson(y, lmbda):
    epsilon = np.finfo(np.float).eps
    y = np.array(y, dtype=float)
    result = y
    if not (isinstance(lmbda, list) or isinstance(lmbda, np.ndarray)):
        lmbda, y = np.broadcast_arrays(lmbda, y)
        lmbda = np.array(lmbda, dtype=float)
    l0 = np.abs(lmbda) > epsilon
    l2 = np.abs(lmbda - 2) > epsilon

    # Inverse
    with warnings.catch_warnings():  # suppress warnings
        warnings.simplefilter("ignore")

        mask = np.where(((y >= 0) & l0) == True)
        result[mask] = np.power(np.multiply(y[mask], lmbda[mask]) + 1, 1 / lmbda[mask]) - 1

        mask = np.where(((y >= 0) & ~l0) == True)
        result[mask] = np.expm1(y[mask])

        mask = np.where(((y < 0) & l2) == True)
        result[mask] = 1 - np.power(np.multiply(-(2 - lmbda[mask]), y[mask]) + 1, 1 / (2 - lmbda[mask]))

        mask = np.where(((y < 0) & ~l2) == True)
        result[mask] = -np.expm1(-y[mask])
    return result


def yeo_johnson(y, l):
    # yeo jhonson transformation function
    yt = np.array(y).copy()
    if (l != 0):
        yt[y >= 0] = ((y[y >= 0] + 1) ** l - 1) / l
    else:
        yt[y >= 0] = np.log(y[y >= 0] + 1)

    if (l != 2):
        yt[y < 0] = -((-y[y < 0] + 1) ** (2 - l) - 1) / (2 - l)
    else:
        yt[y < 0] = -np.log(-y[y < 0] + 1)

    return yt


def yj_quantile(freq, bins, q, dist=norm, **kwargs):
    freq = pd.Series(freq)
    bins = pd.Series(bins)
    rfreq = freq.cumsum() / freq.sum()

    # excluding problematic bins

    msk = (rfreq > 0) & (rfreq < rfreq.iloc[-1])
    rfreq, bins = rfreq[msk], bins[msk]

    # transformation
    qnorm_freq = rfreq.map(lambda x: dist.ppf(x, **kwargs))
    transformation = lambda l: yeojohnson(bins, l)

    # find lambda
    f = lambda l: -stats.pearsonr(transformation(l), qnorm_freq)[0]
    opt_results = minimize_scalar(f, method='bounded', bounds=(0, 2))
    lmbda = opt_results.get("x")

    bins_transormed = transformation(lmbda)
    qnorm_freq = list(qnorm_freq)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x=qnorm_freq, y=bins_transormed)
    q_hat = intercept + slope * dist.ppf(q, **kwargs)
    return inv_yeo_johnson(np.array([q_hat]), lmbda)


@dataclass
class Gamma:

    def sig_mu_to_shape(self, mu, sd):
        return (mu / sd) ** 2

    def sig_mu_to_scale(self, mu, sd):
        rate = (mu / sd ** 2)
        return 1 / rate

    def shift_sd(self, q, tau, sd):
        return (1 / 2) * np.log((q + tau * sd) / (q - tau * sd))


def mixed_normal_quantile(mus, sds, ns, q=0.95, dist=norm, **kwargs):
    assert len(mus) == len(sds) == len(ns), f"length of mus,sds,ns has to be equal"
    N = sum(ns)
    f = lambda x: sum(dist.cdf((x - mu) / sd, **kwargs) * (n / N) for mu, sd, n in zip(mus, sds, ns)) - q
    root = scipy.optimize.brentq(f, -999, 999)
    return root


def mixed_gamma_quantile(shapes, scales, ns, q, **kwargs):
    assert len(shapes) == len(scales) == len(ns), f"length of mus,sds,ns has to be equal"
    N = sum(ns)
    dist = scipy.stats.gamma
    f = lambda x: sum(
        dist.cdf(x, a=shape, scale=scale, **kwargs) * (n / N)
        for shape, scale, n in zip(shapes, scales, ns)
    ) - q
    root = scipy.optimize.brentq(f, -999, 999)
    return root


def get_allocation(n, k) -> list:
    k_range = np.arange(1, k + 1)
    sqewed_range = np.exp(k_range / (k + 1) ** .7)
    c = np.log(n) - np.log(sum(sqewed_range))
    a = (np.exp(c) * sqewed_range[::-1]).astype(int)
    a[0] += (n - sum(a))
    return list(a)


def normed(x):
    sumx = np.sum(x)
    if sumx == 0:
        return np.zeros_like(x)
    return np.array(x) / np.sum(x)


def get_last_first_bin(x1, x2, k=10):
    vals = sorted(list(list(x1) + list(x2)))
    return vals[-1] + np.mean(np.diff(vals[-k:])), vals[0] - np.mean(np.diff(vals[:k]))


def make_str_ok_for_file(s):
    unauthorized_symbols = [':', '.', '-', ' ']
    for symb in unauthorized_symbols:
        s = s.replace(symb, '_')
    return s
