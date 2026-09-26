"""Pytest configuration for the numeric-core regression suite (audit R5.3).

The project uses a deliberately flat module layout (no top-level package;
the Tk mega-class and its mixins assume sibling imports). This conftest
puts ``CODE/`` on ``sys.path`` so the tests can import the flat modules
(``sim_calibration``, ``baselines``, ...) and the ``experiments`` package
exactly as the application and experiment runners do.

Run from ``CODE/``:
    python -m pytest -q
"""

import os
import random
import sys

import numpy as np
import pytest

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)


def _seed_global_streams():
    np.random.seed(0)
    random.seed(0)


# The simulator draws agent decisions from the global numpy stream, and
# some tests leave it in a state that depends on timing (the threaded
# loop runs as many ticks as the wall clock allows). Reseeding before
# every module's fixtures and before every test makes each test's
# realization independent of which tests ran before it, so a subset
# (-k, --lf, one file) sees the same numbers as the full suite.
@pytest.fixture(autouse=True, scope='module')
def _seeded_global_streams_per_module():
    _seed_global_streams()


@pytest.fixture(autouse=True)
def _seeded_global_streams_per_test():
    _seed_global_streams()
