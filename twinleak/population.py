# SPDX-License-Identifier: Apache-2.0
"""Synthetic population generator. No real people, no real records."""
from dataclasses import dataclass
import numpy as np

REGIONS = [f"Region {i:02d}" for i in range(20)]
ZONE_OF_REGION = np.arange(20) // 4          # 5 zones of 4 regions
DECADES = [20, 30, 40, 50, 60, 70, 80, 90]   # decade buckets (18-29 -> 20s, ...)
CONDITIONS = ["diabetes", "hypertension", "asthma", "sickle cell", "HIV",
              "depression", "CKD", "COPD", "arthritis", "epilepsy",
              "heart failure", "thyroid disorder"]
# one typical medication per condition; MED_CLASS groups them coarsely
MEDS = ["metformin", "amlodipine", "salbutamol", "hydroxyurea", "dolutegravir",
        "sertraline", "furosemide", "tiotropium", "meloxicam", "levetiracetam",
        "bisoprolol", "levothyroxine"]
MED_CLASS = np.array([0, 1, 2, 3, 4, 5, 1, 2, 6, 5, 1, 3])  # 7 coarse classes
# base prevalence and how strongly it rises with age
PREV_BASE = np.array([.05, .10, .06, .02, .02, .05, .02, .02, .05, .01, .01, .03])
PREV_AGE = np.array([.20, .45, .00, .00, .00, .00, .10, .15, .30, .00, .12, .06])


@dataclass
class Population:
    age: np.ndarray        # int years
    decade: np.ndarray     # index into DECADES
    sex: np.ndarray        # 0 female, 1 male
    region: np.ndarray     # index into REGIONS
    conds: np.ndarray      # bool (n, 12)
    meds: np.ndarray       # bool (n, 12)
    hba1c: np.ndarray
    sbp: np.ndarray

    @property
    def n(self):
        return len(self.age)


def generate(n: int, seed: int) -> Population:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.gamma(4.0, 9.0, n) + 18, 18, 99).astype(int)
    decade = np.clip(np.where(age < 30, 0, age // 10 - 2), 0, len(DECADES) - 1)
    sex = rng.integers(0, 2, n)
    # uneven regional sizes, like real districts
    rw = rng.dirichlet(np.full(20, 2.0))
    region = rng.choice(20, n, p=rw)
    a = (age - 18) / 80.0
    p = np.clip(PREV_BASE[None, :] + PREV_AGE[None, :] * a[:, None] ** 2, 0, .9)
    conds = rng.random((n, 12)) < p
    meds = conds & (rng.random((n, 12)) < 0.85)      # most treated
    hba1c = np.round(5.3 + conds[:, 0] * rng.normal(2.0, .8, n) + rng.normal(0, .3, n), 1)
    sbp = np.round(118 + conds[:, 1] * rng.normal(22, 8, n) + a * 15 + rng.normal(0, 8, n))
    return Population(age, decade, sex, region, conds, meds, hba1c, sbp)
