# SPDX-License-Identifier: Apache-2.0
"""Atomic claim extraction from twin output text (rule-based baseline).

Each claim is (attribute, value). The project replaces/extends this with an LLM extractor;
the rule-based one keeps the reference pipeline free of paid APIs and fully reproducible.
"""
import re
from .population import REGIONS, DECADES, CONDITIONS, MEDS
from .releases import CLASS_NAMES, BANDS

_P = [
    (re.compile(r"As a (man|woman)"), lambda m: ("sex", 1 if m[1] == "man" else 0)),
    (re.compile(r"(Region \d\d) General"), lambda m: ("region", REGIONS.index(m[1]))),
    (re.compile(r"Clinics in Zone (\d)"), lambda m: ("zone", int(m[1]))),
    (re.compile(r"in their (\d\d)s"), lambda m: ("decade", DECADES.index(int(m[1])))),
    (re.compile(r"adults aged (under 40|40 to 59|60 or over)"), lambda m: ("band", BANDS.index(m[1]))),
    (re.compile(r"take your ([\w/\-]+) medicine"), lambda m: ("medclass", CLASS_NAMES.index(m[1]))),
    (re.compile(r"take your (\w+) tonight"), lambda m: ("med", MEDS.index(m[1]))),
    (re.compile(r"Your ([\w ]+) check-in"), lambda m: ("cond", CONDITIONS.index(m[1]))),
    (re.compile(r"HbA1c averaged ([\d.]+)%"), lambda m: ("hba1c", float(m[1]))),
    (re.compile(r"systolic blood pressure averaged (\d+)"), lambda m: ("sbp", int(m[1]))),
]


def extract(text):
    for pat, f in _P:
        m = pat.search(text)
        if m:
            return [f(m)]
    return []
