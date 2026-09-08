"""macOS kernel sandbox envelope for test-only continuation processes.

The sandbox is inherited by every nested child, including native extension I/O.
It supplements (does not replace) recursive reference checks and audit counters.
"""
from __future__ import annotations
import json
import os
from pathlib import Path


def validate_fixture_root(fixture_root):
    """Reject protected/worktree locations before creating a fixture or sandbox."""
    fixture = Path(fixture_root)
    if not fixture.is_absolute() or not fixture.name.startswith('synthetic_') or '..' in fixture.parts:
        raise ValueError('independent synthetic root required')
    for ancestor in (fixture, *fixture.parents):
        if ancestor.is_symlink():
            raise ValueError('fixture root symlink')
        if (ancestor / '.git').exists():
            raise ValueError('fixture cannot be inside a source worktree')
    allowed = (Path('/private/tmp'), Path('/private/var/folders'))
    if not any(base in fixture.parents for base in allowed):
        raise ValueError('fixture must use independent acceptance temporary storage')
    return fixture


def sandbox_command(command, fixture_root, scientific_root):
    fixture = validate_fixture_root(fixture_root)
    science = Path(scientific_root)
    if not fixture.is_absolute() or not fixture.name.startswith('synthetic_') or '..' in fixture.parts:
        raise ValueError('kernel sandbox requires explicit synthetic root')
    for part in (fixture, *fixture.parents):
        if part.is_symlink():
            raise ValueError('kernel sandbox fixture symlink')
    denied_reads = [Path('/Users/howen/Projects/PPO_MEC/artifacts/experiments'),
                    Path('/Users/howen/Projects/PPO_MEC/data'), science / 'data', science / 'artifacts']
    profile = '(version 1) (allow default) (deny network*) (deny file-write*) '
    profile += '(allow file-write* (subpath ' + json.dumps(str(fixture)) + ') (literal "/dev/null")) '
    for path in denied_reads:
        profile += '(deny file-read* (subpath ' + json.dumps(str(path)) + ')) '
    return ['/usr/bin/sandbox-exec', '-p', profile, *command]


def fixture_environment(fixture_root, scientific_root):
    fixture = validate_fixture_root(fixture_root)
    return dict(os.environ, PYTHONPATH=str(scientific_root), PYTHONDONTWRITEBYTECODE='1',
                PYTHONNOUSERSITE='1', GIT_OPTIONAL_LOCKS='0',
                TMPDIR=str(fixture), MPLCONFIGDIR=str(fixture / '.matplotlib'),
                XDG_CACHE_HOME=str(fixture / '.cache'), TORCH_HOME=str(fixture / '.torch'),
                PPO_MEC_SYNTHETIC_KERNEL_SANDBOX='required')


def verify_kernel_boundary(fixture_root):
    """Verify OS allow/deny with disposable paths in the independent test directory.

    The denied path is a fresh sibling in the acceptance temp directory, never a
    real run or source path. No real input is opened by this boundary probe.
    """
    import uuid
    if os.environ.get('PPO_MEC_SYNTHETIC_KERNEL_SANDBOX') != 'required':
        raise ValueError('synthetic process requires kernel sandbox envelope')
    fixture = validate_fixture_root(fixture_root)
    inside = fixture / ('kernel_probe_' + uuid.uuid4().hex)
    outside = fixture.parent / ('kernel_denied_probe_' + uuid.uuid4().hex)
    inside.write_bytes(b'test-only boundary probe')
    inside.unlink()
    try:
        fd = os.open(outside, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except PermissionError:
        denied = True
    else:
        os.close(fd)
        outside.unlink()
        denied = False
    if not denied:
        raise ValueError('kernel write boundary not enforced')
    return {'status': 'pass', 'inside_write': 'allowed', 'outside_write': 'denied_by_os',
            'outside_path_scope': 'disposable sibling in independent acceptance directory',
            'scope': 'kernel restriction inherited by nested children and native extensions'}
