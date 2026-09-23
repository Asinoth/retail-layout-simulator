"""Where the calibration datasets live on disk: the one search order.

The GUI chooser, the dataset smoke, the figure scripts, the live
diagnostics and the tests all find the real datasets through this module,
so a workbook saved under the name its publisher distributes it under is
found everywhere or nowhere, and an override set once applies to every
entry point.

Search order, per dataset:

  1. an explicit path handed in by the caller (a ``--retail-path`` flag,
     say) -- it must exist, since silently loading some other copy would
     put a different file's SHA-256 into the provenance record;
  2. the dataset's environment variable (``UCI_RETAIL_XLSX``,
     ``OMNICHANNEL_DIR``, ``OPENTRAJ_ETH_OBSMAT``), held to the same rule;
  3. the known file or folder names inside ``datasets_dir()``, which is
     the first existing of ``$RETAIL_DATASETS_DIR``, ``<repo>/DATASETS``
     (the sibling of ``CODE/``), ``<CODE>/DATASETS`` and
     ``<cwd>/DATASETS``.

A miss raises ``FileNotFoundError`` whose message lists every path that
was tried and where the data can be downloaded, so the reader of a GUI
dialog or a failed run has what they need to fix it. None of the three
sources is redistributed with the code.

The module also owns the repository root the search is anchored at, and
``repo_relative`` is how run records name an in-repo location without
carrying the checkout's absolute path (and with it the author's home
folder) into a shipped JSON.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

UCI_URL = 'https://archive.ics.uci.edu/dataset/502/online+retail+ii'
OMNICHANNEL_URL = 'https://github.com/JoyjitBhowmick/Omnichannel-Retail-Datasets'
OPENTRAJ_URL = 'https://github.com/crowdbotp/OpenTraj'

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CODE_DIR)

ENV_DATASETS_DIR = 'RETAIL_DATASETS_DIR'
ENV_UCI = 'UCI_RETAIL_XLSX'
ENV_OMNICHANNEL = 'OMNICHANNEL_DIR'
ENV_OPENTRAJ = 'OPENTRAJ_ETH_OBSMAT'

# In order of preference. The first is the name UCI distributes the
# workbook under; the last is the name an earlier checkout of this project
# saved it under, still accepted so an existing DATASETS/ keeps working.
UCI_NAMES = (
    'online_retail_II.xlsx',
    'Online Retail II.xlsx',
    'online_retail_II.xls',
    'UCI Online Retail II .xlsx.xlsx',
)
# GitHub's "Download ZIP" appends the branch; a git clone does not.
OMNICHANNEL_NAMES = (
    'Omnichannel-Retail-Datasets-main',
    'Omnichannel-Retail-Datasets',
)
OPENTRAJ_TOPS = ('OpenTraj-master', 'OpenTraj')
# The ETH scene's world-coordinate annotation inside an OpenTraj checkout.
ETH_OBSMAT_PARTS = ('datasets', 'ETH', 'seq_eth', 'obsmat.txt')


def _candidate_dirs() -> List[str]:
    """The DATASETS/ locations ``datasets_dir`` considers, in order."""
    out = []
    env = os.environ.get(ENV_DATASETS_DIR)
    if env:
        out.append(env)
    out += [os.path.join(REPO_ROOT, 'DATASETS'),
            os.path.join(CODE_DIR, 'DATASETS'),
            os.path.join(os.getcwd(), 'DATASETS')]
    return out


def datasets_dir() -> Optional[str]:
    """The first existing DATASETS/ folder, or None when there is none.

    ``$RETAIL_DATASETS_DIR`` comes first, then the repository layout
    (``DATASETS/`` beside ``CODE/``), then a ``DATASETS/`` inside ``CODE/``
    as older flat checkouts had it, then one under the working directory."""
    for c in _candidate_dirs():
        if os.path.isdir(c):
            return c
    return None


def _not_found(what: str, tried: Sequence[str], url: str, env: str,
               place: str) -> FileNotFoundError:
    root = datasets_dir() or os.path.join(REPO_ROOT, 'DATASETS')
    lines = [f'{what} not found. Tried:']
    lines += [f'  {p}' for p in tried]
    lines.append(f'Download it from {url} and place {place} in {root}, '
                 f'or set {env} to its path.')
    return FileNotFoundError('\n'.join(lines))


def _find(what: str, path: Optional[str], env: str, candidates: Sequence[str],
          exists, url: str, place: str) -> str:
    """Shared lookup: explicit path, then the environment variable, then
    the candidate names under ``datasets_dir()``.

    ``candidates`` are paths relative to the datasets folder. A path that
    was asked for by name -- explicitly or through ``env`` -- has to exist;
    it is never quietly replaced by a different copy found by the search.
    ``place`` completes the download hint ("place <place> in <folder>").
    """
    if path:
        if exists(path):
            return path
        raise _not_found(what, [f'{path} (given explicitly)'], url, env, place)
    env_path = os.environ.get(env)
    if env_path:
        if exists(env_path):
            return env_path
        raise _not_found(what, [f'{env_path} (from ${env})'], url, env, place)
    root = datasets_dir()
    if root is None:
        tried = [f'{d} (no such folder)' for d in _candidate_dirs()]
        raise _not_found(what, tried, url, env, place)
    tried = []
    for rel in candidates:
        p = os.path.join(root, rel)
        if exists(p):
            return p
        tried.append(p)
    raise _not_found(what, tried, url, env, place)


def uci_workbook(path: Optional[str] = None) -> str:
    """Path to the UCI Online Retail II workbook.

    ``path`` wins when given (and must be an existing file); otherwise
    ``$UCI_RETAIL_XLSX``; otherwise the first of ``UCI_NAMES`` present in
    ``datasets_dir()``."""
    return _find('UCI Online Retail II workbook', path, ENV_UCI, UCI_NAMES,
                 os.path.isfile, UCI_URL, f"it as '{UCI_NAMES[0]}'")


def omnichannel_dir(path: Optional[str] = None) -> str:
    """Path to the Omnichannel Retail bundle (a folder of CSVs).

    ``path`` wins when given (and must be an existing folder); otherwise
    ``$OMNICHANNEL_DIR``; otherwise the first of ``OMNICHANNEL_NAMES``
    present in ``datasets_dir()``."""
    return _find('Omnichannel Retail bundle', path, ENV_OMNICHANNEL,
                 OMNICHANNEL_NAMES, os.path.isdir, OMNICHANNEL_URL,
                 f"the repository as '{OMNICHANNEL_NAMES[0]}'")


def opentraj_eth_obsmat(path: Optional[str] = None) -> str:
    """Path to the OpenTraj ETH ``obsmat.txt`` (the seq_eth scene).

    ``path`` wins when given (and must be an existing file); otherwise
    ``$OPENTRAJ_ETH_OBSMAT``; otherwise
    ``<top>/datasets/ETH/seq_eth/obsmat.txt`` in ``datasets_dir()``, with
    ``<top>`` either ``OpenTraj-master`` (the ZIP download) or ``OpenTraj``
    (a clone)."""
    rels = [os.path.join(top, *ETH_OBSMAT_PARTS) for top in OPENTRAJ_TOPS]
    return _find('OpenTraj ETH obsmat.txt', path, ENV_OPENTRAJ, rels,
                 os.path.isfile, OPENTRAJ_URL,
                 f"the repository as '{OPENTRAJ_TOPS[0]}'")


def repo_relative(path: str) -> str:
    """``path`` relative to the repository root, with forward slashes, when
    it lies inside the checkout; its absolute path otherwise.

    Run records use this for the directories they wrote to: an in-repo
    location then reads the same on every machine, and a shipped JSON does
    not carry the absolute path of the checkout it was produced in."""
    ap = os.path.abspath(path)
    root = os.path.abspath(REPO_ROOT)
    try:
        rel = os.path.relpath(ap, root)
    except ValueError:              # another drive on Windows
        return ap
    if rel == os.curdir:
        return '.'
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return ap
    return rel.replace(os.sep, '/')
