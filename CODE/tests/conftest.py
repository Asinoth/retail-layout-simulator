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
import sys

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
