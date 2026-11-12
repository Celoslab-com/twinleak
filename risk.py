# SPDX-License-Identifier: Apache-2.0
"""Recipient linkage model and the person-side disclosure meter.

A recipient holds an auxiliary table about everyone (prior knowledge level K):
  K1  public register: sex, region, age decade
  K2  K1 + medication list (e.g. a pharmacy dataset)
  K3  K2 + diagnosed conditions
It scores every candidate by the likelihood of the extracted claims and forms a posterior.

The meter runs on the person's own device. It never sees the real population; it uses a
public synthetic reference population of the same size class to estimate how many others
would match the person's accumulated releases.
"""
import numpy as np
from .population import ZONE_OF_REGION, MED_CLASS, REGIONS, DECADES, MEDS, CONDITIONS
from .releases import band_of_decade

KNOWS = {
    "K1": {"sex", "region", "zone", "decade", "band"},
    "K2": {"sex", "region", "zone", "decade", "band", "med", "medclass"},
    "K3": {"sex", "region", "zone", "decade", "band", "med", "medclass", "cond"},
}
_K = {"sex": 2, "region": len(REGIONS), "decade": len(DECADES), "med": len(MEDS),
      "cond": len(CONDITIONS)}


def claim_loglik(pop, attr, val, mode, eps):
    """log P(observed claim | candidate) for every member of pop."""
    if attr == "sex":
        match = pop.sex == val
    elif attr == "region":
        match = pop.region == val
    elif attr == "zone":
        match = ZONE_OF_REGION[pop.region] == val
    elif attr == "decade":
        match = pop.decade == val
    elif attr == "band":
        match = np.vectorize(band_of_decade)(pop.decade) == val
    elif attr == "med":
        match = pop.meds[:, val]
    elif attr == "medclass":
        match = pop.meds[:, MED_CLASS == val].any(axis=1)
    elif attr == "cond":
        match = pop.conds[:, val]
    else:
        return np.zeros(pop.n)
    if mode == "dp" and attr in _K:
        k = _K[attr]
        p = np.exp(eps) / (np.exp(eps) + k - 1)
        return np.where(match, np.log(p), np.log((1 - p) / (k - 1)))
    return np.where(match, 0.0, -np.inf)


def posterior_of(loglik, idx):
    m = np.max(loglik)
    if not np.isfinite(m):
        return 0.0
    w = np.exp(loglik - m)
    return float(w[idx] / w.sum())


def meter_estimate(self_ll, ref_loglik, n_population):
    """Person-side estimate of P(recipient singles me out).

    self_ll: log-likelihood of the person's own record (they know it).
    ref_loglik: log-likelihoods over the public reference population.
    Expected matching weight of the other n-1 people = (n-1) * mean_ref(L).
    """
    if not np.isfinite(self_ll):
        return 0.0
    m = max(self_ll, np.max(ref_loglik))
    ws = np.exp(self_ll - m)
    others = (n_population - 1) * np.mean(np.exp(ref_loglik - m))
    return float(ws / (ws + others))
