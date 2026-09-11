"""Single writer: persistent-inode flock, no PID/time-based lock removal."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import uuid

from .identity import ContinuationError, absolute_path, canonical, digest, within


def process_identity(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], text=True,
                            capture_output=True, check=False)
    return {"host": socket.gethostname(), "pid": pid,
            "started": result.stdout.strip() if result.returncode == 0 else None}


def writer_lock_path(run_root):
    root = absolute_path(str(run_root))
    return root.parent / ".continuation_locks" / (digest(str(root)) + ".lock")


class SingleWriter:
    """Authorization is checked before open and again after acquiring the lock.

    A crash leaves the owner record. Recovery needs a fresh signed contract naming
    its full hash plus externally established quiescence. The kernel lock must be
    obtainable and the exact previous owner must be gone. Inode is never removed.
    """
    def __init__(self, run_root, executor_sha256, authorize, recovery_owner_sha256=None, *, allow_missing_root=False):
        self.root = Path(run_root)
        self.executor_sha256 = executor_sha256
        self.authorize = authorize
        self.recovery_owner_sha256 = recovery_owner_sha256
        self.fd = None
        self.allow_missing_root = allow_missing_root

    def __enter__(self):
        self.authorize()
        path = absolute_path(str(writer_lock_path(self.root)))
        if not self.root.is_dir() and not self.allow_missing_root:
            raise ContinuationError("continuation may not create a second run")
        if self.allow_missing_root:
            # Evaluation bootstrap requires an existing, writable parent and a
            # usable host identity before creating any coordination/run state.
            if not self.root.parent.is_dir() or not os.access(self.root.parent, os.W_OK | os.X_OK):
                raise ContinuationError("evaluation parent is not writable")
            if not process_identity(os.getpid())["started"]:
                raise ContinuationError("evaluation process identity permission unavailable")
        path.parent.mkdir(exist_ok=True)
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.authorize()
            previous_bytes = os.read(self.fd, 65537)
            if len(previous_bytes) > 65536:
                raise ContinuationError("lock owner record oversized")
            if previous_bytes:
                previous = json.loads(previous_bytes)
                if previous.get("state") != "released":
                    if digest(previous) != self.recovery_owner_sha256:
                        raise ContinuationError("crash recovery requires exact approved previous owner")
                    observed = process_identity(previous["process"]["pid"])
                    if observed["host"] != previous["process"]["host"] or observed == previous["process"]:
                        raise ContinuationError("previous owner not proven gone")
            self.owner = {"version": "1.0.0", "state": "held", "nonce": uuid.uuid4().hex,
                          "process": process_identity(os.getpid()), "executor_sha256": self.executor_sha256}
            self._write(self.owner)
            return self
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def _write(self, value):
        os.lseek(self.fd, 0, os.SEEK_SET)
        data = canonical(value) + b"\n"
        if os.write(self.fd, data) != len(data):
            raise ContinuationError("short lock record write")
        os.ftruncate(self.fd, len(data))
        os.fsync(self.fd)

    def __exit__(self, exc_type, exc, tb):
        try:
            # Exceptions retain the crash/quiescence requirement. Expiry never
            # invalidates committing an already admitted transaction.
            if exc_type is None:
                self._write({**self.owner, "state": "released"})
        finally:
            os.close(self.fd)
            self.fd = None
