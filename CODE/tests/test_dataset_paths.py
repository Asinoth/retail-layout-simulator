"""Dataset discovery: one search order, and a miss that says what it tried.

Every test builds its own DATASETS/ under ``tmp_path`` and points the
module's anchors (repository root, CODE/, working directory) at empty
temporary folders, so nothing here depends on -- or can be satisfied by --
the real datasets of the checkout the suite runs in.
"""
import os

import pytest

import dataset_paths as DP


ENV_VARS = (DP.ENV_DATASETS_DIR, DP.ENV_UCI, DP.ENV_OMNICHANNEL,
            DP.ENV_OPENTRAJ)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """No override set, and every default search location empty."""
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    repo = tmp_path / 'repo'
    code = repo / 'CODE'
    code.mkdir(parents=True)
    cwd = tmp_path / 'cwd'
    cwd.mkdir()
    monkeypatch.setattr(DP, 'REPO_ROOT', str(repo))
    monkeypatch.setattr(DP, 'CODE_DIR', str(code))
    monkeypatch.chdir(cwd)
    return tmp_path


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')
    return str(path)


def _datasets(isolated, monkeypatch):
    """A DATASETS/ folder the search will use, via the override."""
    d = isolated / 'DATASETS'
    d.mkdir()
    monkeypatch.setenv(DP.ENV_DATASETS_DIR, str(d))
    return d


# --- datasets_dir -------------------------------------------------------------

def test_no_datasets_folder_is_none(isolated):
    assert DP.datasets_dir() is None


def test_repo_layout_found_before_code_and_cwd(isolated):
    repo_ds = isolated / 'repo' / 'DATASETS'
    code_ds = isolated / 'repo' / 'CODE' / 'DATASETS'
    cwd_ds = isolated / 'cwd' / 'DATASETS'
    for d in (cwd_ds, code_ds, repo_ds):
        d.mkdir()
    assert DP.datasets_dir() == str(repo_ds)
    repo_ds.rmdir()
    assert DP.datasets_dir() == str(code_ds)
    code_ds.rmdir()
    assert os.path.samefile(DP.datasets_dir(), cwd_ds)


def test_env_datasets_dir_first(isolated, monkeypatch):
    (isolated / 'repo' / 'DATASETS').mkdir()
    d = _datasets(isolated, monkeypatch)
    assert DP.datasets_dir() == str(d)


# --- uci_workbook -------------------------------------------------------------

def test_explicit_path_wins(isolated, monkeypatch):
    d = _datasets(isolated, monkeypatch)
    _touch(d / DP.UCI_NAMES[0])
    monkeypatch.setenv(DP.ENV_UCI, _touch(isolated / 'env' / 'env.xlsx'))
    explicit = _touch(isolated / 'elsewhere' / 'mine.xlsx')
    assert DP.uci_workbook(explicit) == explicit


def test_missing_explicit_path_raises_rather_than_searching(isolated,
                                                            monkeypatch):
    d = _datasets(isolated, monkeypatch)
    _touch(d / DP.UCI_NAMES[0])
    missing = str(isolated / 'nope.xlsx')
    with pytest.raises(FileNotFoundError) as exc:
        DP.uci_workbook(missing)
    assert missing in str(exc.value)


def test_env_var_wins_over_search(isolated, monkeypatch):
    d = _datasets(isolated, monkeypatch)
    _touch(d / DP.UCI_NAMES[0])
    env_path = _touch(isolated / 'env' / 'from_env.xlsx')
    monkeypatch.setenv(DP.ENV_UCI, env_path)
    assert DP.uci_workbook() == env_path


def test_env_var_naming_a_missing_file_raises(isolated, monkeypatch):
    d = _datasets(isolated, monkeypatch)
    _touch(d / DP.UCI_NAMES[0])
    missing = str(isolated / 'env' / 'gone.xlsx')
    monkeypatch.setenv(DP.ENV_UCI, missing)
    with pytest.raises(FileNotFoundError) as exc:
        DP.uci_workbook()
    assert missing in str(exc.value) and DP.ENV_UCI in str(exc.value)


def test_canonical_name_preferred_when_both_exist(isolated, monkeypatch):
    d = _datasets(isolated, monkeypatch)
    legacy = _touch(d / 'UCI Online Retail II .xlsx.xlsx')
    canonical = _touch(d / 'online_retail_II.xlsx')
    assert DP.uci_workbook() == canonical != legacy


def test_legacy_name_still_found(isolated, monkeypatch):
    d = _datasets(isolated, monkeypatch)
    legacy = _touch(d / 'UCI Online Retail II .xlsx.xlsx')
    assert DP.uci_workbook() == legacy


def test_found_in_repo_layout_without_any_override(isolated):
    wb = _touch(isolated / 'repo' / 'DATASETS' / 'online_retail_II.xlsx')
    assert DP.uci_workbook() == wb


def test_error_lists_every_path_tried_and_the_source(isolated, monkeypatch):
    d = _datasets(isolated, monkeypatch)
    with pytest.raises(FileNotFoundError) as exc:
        DP.uci_workbook()
    msg = str(exc.value)
    for name in DP.UCI_NAMES:
        assert os.path.join(str(d), name) in msg
    assert DP.UCI_URL in msg and DP.ENV_UCI in msg


def test_error_without_a_datasets_folder_lists_the_folders(isolated):
    with pytest.raises(FileNotFoundError) as exc:
        DP.uci_workbook()
    msg = str(exc.value)
    assert os.path.join(str(isolated / 'repo'), 'DATASETS') in msg
    assert os.path.join(str(isolated / 'repo' / 'CODE'), 'DATASETS') in msg
    assert DP.UCI_URL in msg


# --- omnichannel_dir / opentraj_eth_obsmat -----------------------------------

def test_omnichannel_clone_name_accepted(isolated, monkeypatch):
    d = _datasets(isolated, monkeypatch)
    (d / 'Omnichannel-Retail-Datasets').mkdir()
    assert DP.omnichannel_dir() == str(d / 'Omnichannel-Retail-Datasets')
    (d / 'Omnichannel-Retail-Datasets-main').mkdir()
    assert DP.omnichannel_dir() == str(d / 'Omnichannel-Retail-Datasets-main')


def test_omnichannel_env_and_error(isolated, monkeypatch):
    d = _datasets(isolated, monkeypatch)
    with pytest.raises(FileNotFoundError) as exc:
        DP.omnichannel_dir()
    assert DP.OMNICHANNEL_URL in str(exc.value)
    bundle = isolated / 'bundle'
    bundle.mkdir()
    monkeypatch.setenv(DP.ENV_OMNICHANNEL, str(bundle))
    assert DP.omnichannel_dir() == str(bundle)
    # A file is not a bundle folder.
    with pytest.raises(FileNotFoundError):
        DP.omnichannel_dir(_touch(d / 'demand.csv'))


def test_opentraj_zip_and_clone_layouts(isolated, monkeypatch):
    d = _datasets(isolated, monkeypatch)
    clone = _touch(d.joinpath('OpenTraj', *DP.ETH_OBSMAT_PARTS))
    assert DP.opentraj_eth_obsmat() == clone
    zipped = _touch(d.joinpath('OpenTraj-master', *DP.ETH_OBSMAT_PARTS))
    assert DP.opentraj_eth_obsmat() == zipped


def test_opentraj_env_wins_and_error_names_source(isolated, monkeypatch):
    d = _datasets(isolated, monkeypatch)
    with pytest.raises(FileNotFoundError) as exc:
        DP.opentraj_eth_obsmat()
    assert DP.OPENTRAJ_URL in str(exc.value)
    _touch(d.joinpath('OpenTraj-master', *DP.ETH_OBSMAT_PARTS))
    env_path = _touch(isolated / 'eth' / 'obsmat.txt')
    monkeypatch.setenv(DP.ENV_OPENTRAJ, env_path)
    assert DP.opentraj_eth_obsmat() == env_path


# --- repo_relative ------------------------------------------------------------

def test_repo_relative_inside_uses_forward_slashes(isolated):
    inside = isolated / 'repo' / 'CODE' / 'experiments' / 'results'
    assert DP.repo_relative(str(inside)) == 'CODE/experiments/results'
    assert DP.repo_relative(str(isolated / 'repo')) == '.'


def test_repo_relative_outside_stays_absolute(isolated):
    outside = isolated / 'elsewhere' / 'figs'
    assert DP.repo_relative(str(outside)) == os.path.abspath(str(outside))
    sibling = isolated / 'repo-other'
    assert DP.repo_relative(str(sibling)) == os.path.abspath(str(sibling))
