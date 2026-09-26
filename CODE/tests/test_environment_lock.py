"""The resolved environment every sidecar records, and its lock file.

``requirements.txt`` pins the direct dependencies only; the lock file at
the repository root pins everything they pull in, and each sidecar records
the resolved closure as the run saw it plus the lock file's hash and
whether the environment matched it.

The lock is resolved on one platform. Pins with an environment marker
apply only where the marker holds (pip skips the rest), so the check
evaluates them; and the test that the running environment IS the lock
skips, saying why, wherever the environment was not installed from it
(another platform or Python, or ``make deps``, which installs the direct
pins only) -- the format and the marker handling are checked everywhere.
"""

import json
import os
import re

import pytest

import environment_lock as EL
from experiments._common import lock_record, package_versions, write_sidecar


def test_closure_holds_the_direct_dependencies_and_what_they_require():
    closure = EL.dependency_closure()
    for pkg in EL.DIRECT_DEPENDENCIES:
        assert pkg in closure
    # matplotlib and pandas pull these in; a five-package record missed them
    for pkg in ('contourpy', 'pillow', 'python-dateutil', 'pytz', 'tzdata',
                'et-xmlfile'):
        assert pkg in closure, pkg
    assert list(closure) == sorted(closure)


def test_the_shipped_lock_is_well_formed():
    """Every pin is ``name==version`` with an optional marker pip can parse,
    and the header names the platform it was resolved on."""
    from packaging.markers import Marker
    assert os.path.exists(EL.LOCK_PATH)
    header = EL.lock_header()
    assert header['python'] and header['system'] and header['machine']
    pins = 0
    with open(EL.LOCK_PATH, encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\r\n')
            if not line or line.startswith('#'):
                continue
            req, _, marker = line.partition(';')
            assert re.fullmatch(r'[a-z0-9][a-z0-9.-]*==[^\s;]+', req.strip()), line
            if marker.strip():
                Marker(marker.strip())
            pins += 1
    every = EL.read_lock(evaluate_markers=False)
    assert len(every) == pins
    for pkg in EL.DIRECT_DEPENDENCIES:
        assert pkg in every
    rec = EL.lock_record()
    assert rec['path'] == 'requirements-lock.txt'
    assert len(rec['sha256']) == 64
    assert rec['resolved_on'] == header


def test_the_shipped_lock_is_this_environment():
    """Strict only where the environment was installed from the lock, on
    the platform it was resolved on."""
    rec = EL.lock_record()
    if not rec['same_platform']:
        pytest.skip(f"lock resolved on {rec['resolved_on']}, running on "
                    f"{EL.this_platform()}")
    if not rec['matches_environment']:
        pytest.skip("environment not installed from requirements-lock.txt: "
                    + ', '.join(f"{m['package']} {m['installed']} (lock "
                                f"{m['locked']})" for m in rec['mismatches']))
    pinned = EL.read_lock()
    closure = EL.dependency_closure()
    assert set(pinned) == {k for k, v in closure.items() if v['version']}
    assert rec['mismatches'] == [] and rec['unpinned'] == [] and rec['exact']


def test_a_pin_whose_marker_is_false_here_is_not_required(tmp_path):
    """The lock's ``colorama ; sys_platform == "win32"`` on Linux: pip does
    not install it there, so an exact install from the lock must still
    match. Markers are evaluated for this interpreter, as pip does."""
    ver = EL.dependency_closure()['numpy']['version']
    lock = tmp_path / 'requirements-lock.txt'
    lock.write_text('# Python 3.13.2 on Windows AMD64\n'
                    f'numpy=={ver}\n'
                    'colorama==0.4.6 ; sys_platform == "no-such-platform"\n'
                    'notapkg==1.0 ; python_version < "2"\n'
                    'alsonot==2.0 ; sys_platform != "no-such-platform" '
                    'and python_version < "2"\n')
    assert EL.read_lock(str(lock)) == {'numpy': ver}
    assert set(EL.read_lock(str(lock), evaluate_markers=False)) == {
        'numpy', 'colorama', 'notapkg', 'alsonot'}
    rec = EL.lock_record(str(lock))
    assert rec['matches_environment'] is True and rec['mismatches'] == []
    # A marker that holds here is required like any other pin.
    lock.write_text(f'numpy=={ver}\nnotapkg==1.0 ; python_version >= "3"\n')
    rec = EL.lock_record(str(lock))
    assert [m['package'] for m in rec['mismatches']] == ['notapkg']
    assert EL.marker_applies(None) and EL.marker_applies('')


def test_lock_header_and_platform(tmp_path):
    lock = tmp_path / 'l.txt'
    lock.write_text('# Python 3.10.1 on Linux x86_64\nnumpy==1.0\n')
    assert EL.lock_header(str(lock)) == {'python': '3.10.1',
                                         'system': 'Linux',
                                         'machine': 'x86_64'}
    rec = EL.lock_record(str(lock))
    assert rec['same_platform'] == (EL.this_platform() == {
        'python': '3.10.1', 'system': 'Linux', 'machine': 'x86_64'})
    # Everything this environment's closure holds beyond numpy is unpinned.
    assert 'scipy' in rec['unpinned'] and not rec['exact']


def test_the_lock_hash_ignores_line_ends(tmp_path):
    """A checkout that turns LF into CRLF records the same hash."""
    lf, crlf = tmp_path / 'lf.txt', tmp_path / 'crlf.txt'
    text = EL.lock_text()
    lf.write_bytes(text.encode())
    crlf.write_bytes(text.replace('\n', '\r\n').encode())
    assert EL.lock_record(str(lf))['sha256'] == EL.lock_record(str(crlf))['sha256']
    assert (EL.lock_record(str(crlf))['matches_environment']
            == EL.lock_record()['matches_environment'])


def test_lock_record_names_what_drifted(tmp_path):
    lock = tmp_path / 'requirements-lock.txt'
    lock.write_text('numpy==0.0.1\nscipy==%s\n'
                    % EL.dependency_closure()['scipy']['version'])
    rec = EL.lock_record(str(lock))
    assert rec['matches_environment'] is False
    assert [m['package'] for m in rec['mismatches']] == ['numpy']
    assert EL.lock_record(str(tmp_path / 'missing.txt'))['sha256'] is None


def test_sidecar_records_the_full_closure_and_the_lock(tmp_path):
    path = write_sidecar(str(tmp_path), {'experiment': 'lock_test'})
    side = json.load(open(path))
    assert side['packages'] == package_versions()
    assert len(side['packages']) > 5
    assert side['requirements_lock'] == lock_record()
    # The machine, so a family's wall time can be read against it.
    assert side['hardware']['cpu_count'] >= 1
    assert set(side['hardware']) == {'cpu_count', 'processor', 'machine',
                                     'ram_gib'}
