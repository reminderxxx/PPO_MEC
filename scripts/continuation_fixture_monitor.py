"""Test-only process monitor: fixture writes/inputs, dispatch and rollout counters.

Install inside every synthetic Python child. This is acceptance instrumentation,
not a production approval mechanism. Violations raise before the audited operation.
"""
from __future__ import annotations

import os
import atexit
from pathlib import Path
import sys


class FixtureMonitor:
    def __init__(self, *, fixture_root, scientific_root, python_executable,
                 implementation_root, forbidden_roots):
        self.fixture = Path(fixture_root)
        self.scientific = Path(scientific_root)
        self.implementation = Path(implementation_root)
        self.python = os.path.abspath(python_executable)
        self.forbidden = tuple(Path(path) for path in forbidden_roots)
        if not self.fixture.is_absolute() or not self.fixture.name.startswith('synthetic_'):
            raise ValueError('monitor requires explicit synthetic fixture root')
        self.counts = {'synthetic_child_dispatch_count': 0, 'scientific_rollout_count': 0,
                       'real_v16_dispatch_count': 0, 'real_v16_write_count': 0,
                       'fixture_write_operations': 0, 'observed_scientific_calls': 0,
                       'blocked_operations': 0}
        self.events = []
        self.scientific_calls = {}
        self._profile_sources = {}
        self.previous_profile = sys.getprofile()
        self.enabled = False
        self.busy = False

    def _path(self, value):
        if isinstance(value, int) or value is None:
            return None
        try:
            raw = os.fsdecode(value)
        except TypeError:
            raise ValueError('unrecognized audited path')
        path = Path(os.path.abspath(raw))
        # Check lexical input before normalization hides an escape.
        if '..' in Path(raw).parts:
            self._block('parent traversal: ' + raw)
        for part in (path, *path.parents):
            if part.is_symlink():
                self._block('symlink audited path: ' + str(part))
        return path

    def _block(self, detail):
        self.counts['blocked_operations'] += 1
        self.events.append({'kind': 'blocked', 'detail': detail})
        raise ValueError('synthetic boundary: ' + detail)

    def _within(self, path, root):
        return path == root or root in path.parents

    def write(self, value):
        path = self._path(value)
        if path == Path('/dev/null'):
            return
        if path is None:
            # Only stdout/stderr or already opened, audited file descriptors.
            return
        if not self._within(path, self.fixture):
            if any(self._within(path, root) for root in self.forbidden):
                self.events.append({'kind': 'blocked_real_write_attempt', 'path': str(path)})
            self._block('write outside fixture: ' + str(path))
        self.counts['fixture_write_operations'] += 1

    def read(self, value):
        path = self._path(value)
        if path is None:
            return
        if any(self._within(path, root) for root in self.forbidden):
            self._block('real data/result/weight read: ' + str(path))
        # Source and dependency imports are allowed; scientific inputs are copied
        # into the fixture and recursively checked by the dispatcher separately.

    def audit(self, event, args):
        if not self.enabled or self.busy:
            return
        self.busy = True
        try:
            if event == 'open':
                path, mode, flags = args
                writable = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                    isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
                (self.write if writable else self.read)(path)
            elif event in {'os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.utime', 'os.truncate'}:
                self.write(args[0])
            elif event in {'os.rename', 'os.replace'}:
                self.write(args[0]); self.write(args[1])
            elif event == 'os.symlink':
                self._block('symlink creation is not a fixture publication operation')
            elif event == 'os.link':
                self.write(args[0]); self.write(args[1])
            elif event == 'subprocess.Popen':
                executable, argv, cwd, env = args
                if not isinstance(argv, (list, tuple)):
                    self._block('shell dispatch')
                if any(any(str(root) in str(arg) for root in self.forbidden) for arg in argv):
                    self.events.append({'kind': 'blocked_real_dispatch_attempt', 'argv': list(argv)})
                    self._block('real reference in child argv')
                if str(executable) == 'git' and list(argv[:2]) == ['git', 'rev-parse']:
                    if any(str(arg) not in {'git', 'rev-parse', 'HEAD', '--show-toplevel'} for arg in argv):
                        self._block('unapproved Git observation')
                    self.events.append({'kind': 'readonly_git_observation', 'argv': list(argv)})
                    return
                if os.path.abspath(str(executable)) != self.python:
                    self._block('unexpected child interpreter/executable')
                if cwd is not None and Path(cwd) != self.scientific:
                    self._block('child cwd differs from pinned scientific source')
                effective_env = os.environ if env is None else env
                if effective_env.get('PYTHONPATH') != str(self.scientific):
                    self._block('unbound child environment')
                self.counts['synthetic_child_dispatch_count'] += 1
                self.events.append({'kind': 'child_dispatch', 'argv': list(argv),
                                    'cwd': str(cwd), 'pythonpath': effective_env['PYTHONPATH'], 'explicit_env': env is not None})
            elif event in {'os.system', 'os.exec', 'os.posix_spawn', 'socket.connect'}:
                self._block('unaudited process/network operation: ' + event)
        finally:
            self.busy = False

    def profile(self, frame, event, arg):
        if event != 'call' or not self.enabled:
            return
        module = frame.f_globals.get('__name__', '') or ''
        if not module.startswith(('src.', 'scripts.')):
            return
        self.counts['observed_scientific_calls'] += 1
        name = frame.f_code.co_name
        key = module + '.' + name
        self.scientific_calls[key] = self.scientific_calls.get(key, 0) + 1
        cache_key = (frame.f_code, module, frame.f_globals.get('__file__'))
        if cache_key not in self._profile_sources:
            source = Path(frame.f_code.co_filename)
            if frame.f_code.co_filename == '<string>' and (name in {'__init__', '__repr__', '__eq__', '__hash__', '__setattr__', '__delattr__'} or (frame.f_back and frame.f_back.f_globals.get('__name__') == 'dataclasses')):
                source = Path(frame.f_globals.get('__file__', ''))
            if self.scientific not in source.parents:
                self._block('current/shadow scientific source: ' + str(source))
            self._profile_sources[cache_key] = True
        if (module.startswith('src.envs.') and name in {'step', 'reset'}) or (
            'rollout' in name or name in {'run_real_episode', 'train', 'learn'}):
            self.counts['scientific_rollout_count'] += 1
            self._block('scientific rollout/training called: ' + module + '.' + name)
        if module == 'scripts.run_typed_model_cache_formal_protocol' and name == 'main':
            self._block('original public runner main cannot be called')

    def install(self):
        sys.addaudithook(self.audit)
        self.enabled = True
        sys.setprofile(self.profile)
        atexit.register(self.close)
        return self

    def close(self):
        self.enabled = False
        sys.setprofile(self.previous_profile)

    def report(self):
        return {'counts': dict(self.counts), 'scientific_calls': dict(self.scientific_calls), 'events': list(self.events),
                'monitor': 'Python audit hook + scientific call profiler',
                'fixture_root': str(self.fixture), 'scientific_root': str(self.scientific)}
