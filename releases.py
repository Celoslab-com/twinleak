# SPDX-License-Identifier: Apache-2.0
"""A digital twin's outputs to a recipient, as free text, under three sanitisation modes.

raw         - the twin says what it knows
generalise  - quasi-identifiers coarsened (region->zone, decade->band, drug->class)
dp          - each categorical mention passes through k-ary randomised response (epsilon per release)
"""
import numpy as np
from .population import (REGIONS, DECADES, CONDITIONS, MEDS, MED_CLASS, ZONE_OF_REGION)

CLASS_NAMES = ["glucose-lowering", "cardiovascular", "respiratory", "haematology/endocrine",
               "antiretroviral", "neuro/psychiatric", "anti-inflammatory"]
BANDS = ["under 40", "40 to 59", "60 or over"]
KINDS = ["sex", "region", "decade", "med", "cond", "lab"]
KIND_P = np.array([.10, .15, .15, .25, .15, .20])


def band_of_decade(d):
    return 0 if d <= 1 else (1 if d <= 3 else 2)


def _rr(rng, true_idx, k, eps):
    """k-ary randomised response; returns (reported index, p_keep)."""
    p = np.exp(eps) / (np.exp(eps) + k - 1)
    if rng.random() < p:
        return true_idx, p
    others = [i for i in range(k) if i != true_idx]
    return int(rng.choice(others)), p


def make_releases(pop, i, n_rel, mode, rng, eps=1.0):
    """Return list of free-text strings for person i."""
    out = []
    my_meds = np.flatnonzero(pop.meds[i])
    my_conds = np.flatnonzero(pop.conds[i])
    while len(out) < n_rel:
        kind = rng.choice(KINDS, p=KIND_P)
        if kind == "sex":
            v = int(pop.sex[i])
            if mode == "dp":
                v, _ = _rr(rng, v, 2, eps)
            out.append(f"As a {'man' if v else 'woman'}, your screening schedule differs.")
        elif kind == "region":
            v = int(pop.region[i])
            if mode == "generalise":
                out.append(f"Clinics in Zone {ZONE_OF_REGION[v]} have open slots.")
                continue
            if mode == "dp":
                v, _ = _rr(rng, v, len(REGIONS), eps)
            out.append(f"Nearest clinic: {REGIONS[v]} General Hospital.")
        elif kind == "decade":
            v = int(pop.decade[i])
            if mode == "generalise":
                out.append(f"Guidance for adults aged {BANDS[band_of_decade(v)]} applies.")
                continue
            if mode == "dp":
                v, _ = _rr(rng, v, len(DECADES), eps)
            out.append(f"For people in their {DECADES[v]}s, aim for 150 active minutes a week.")
        elif kind == "med":
            if len(my_meds) == 0:
                continue
            v = int(rng.choice(my_meds))
            if mode == "generalise":
                out.append(f"Reminder: take your {CLASS_NAMES[MED_CLASS[v]]} medicine tonight.")
                continue
            if mode == "dp":
                v, _ = _rr(rng, v, len(MEDS), eps)
            out.append(f"Reminder: take your {MEDS[v]} tonight.")
        elif kind == "cond":
            if len(my_conds) == 0:
                continue
            v = int(rng.choice(my_conds))
            if mode == "dp":
                v, _ = _rr(rng, v, len(CONDITIONS), eps)
            out.append(f"Your {CONDITIONS[v]} check-in is due.")
        else:  # lab figure: sensitive content, not a linkage key
            if rng.random() < .5:
                out.append(f"Your HbA1c averaged {pop.hba1c[i]:.1f}% this quarter.")
            else:
                out.append(f"Your systolic blood pressure averaged {int(pop.sbp[i])} mmHg.")
    return out
