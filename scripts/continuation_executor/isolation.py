"""Fail-closed recursive synthetic input/write scope checks."""
from __future__ import annotations

import csv
from pathlib import Path
import re
import sys

from .identity import ContinuationError, absolute_path, read_json, within, strict_json

_PATH_KEY = re.compile(r"(^|_)(path|paths|root|directory|file|files|checkpoint|manifest|descriptor|registry|resource)(_|$)")


class FixtureScope:
    def __init__(self, root, run_root, scientific_root, python):
        self.root = absolute_path(root)
        self.run_root = within(run_root, self.root)
        self.scientific_root = absolute_path(scientific_root)
        self.python = str(Path(python).absolute())
        if self.run_root == self.root or not self.run_root.name.startswith("synthetic_"):
            raise ContinuationError("fixture run identity")
        self.checked_files = {}

    def path(self, value, *, source=False):
        if source:
            p = absolute_path(value)
            if self.root not in p.parents:
                within(value, self.scientific_root)
            if p.suffix != ".py" or not p.is_file():
                raise ContinuationError("only scientific Python source may be external")
            return p
        return within(value, self.root)

    def value(self, value, key="", *, blueprint=False):
        if isinstance(value, dict):
            for name, item in value.items():
                # These fields contain prohibitions/historical declarations, not
                # executable resource inputs. The native protocol validates them.
                if name in {"supersession", "forbidden_project_source_roots", "invalid_run_roots", "invalid_g14c_v3_run_root"}:
                    continue
                self.value(item, str(name), blueprint=blueprint or name in {"command_templates", "default_expansion_context", "coordinates", "matrix_contexts"})
        elif isinstance(value, (list, tuple)):
            if key in {"commands", "command", "argv"} and value and not blueprint:
                if isinstance(value[0], (list, tuple)):
                    for command in value:
                        self.argv(list(command))
                else:
                    self.argv(list(value))
            else:
                for item in value:
                    self.value(item, key, blueprint=blueprint)
        elif isinstance(value, str):
            if blueprint and ("{" in value or value.startswith("/ABSOLUTE/")):
                return  # All actually dispatched expansions are checked by argv.
            if value == self.python and key in {"python_executable", "resolved_python_absolute_path"}:
                return
            if value == str(self.scientific_root) and key in {"clean_worktree_root", "repository_root"}:
                return
            if key == "virtual_environment_root" and value == sys.prefix:
                return
            if key == "site_packages_paths" and value in sys.path and "site-packages" in Path(value).parts:
                return
            if key == "serialized_state_ref" and value.startswith("controlled://"):
                return  # Typed catalog logical fixture reference, never a URI resolver.
            if "://" in value:
                raise ContinuationError("external URI in synthetic input")
            if value.startswith("/"):
                self.path(value)
            elif _PATH_KEY.search(key) and value:
                if ".." in Path(value).parts or value.startswith("~"):
                    raise ContinuationError("relative path escape")
                self.path(str(self.root / value))
            # Catch paths hidden inside nested argv strings, CSV lists, or options.
            for token in re.findall(r"(?:^|[=,\s])(/[^,\s]+)", value):
                self.path(token)

    def input_file(self, path):
        p = self.path(str(path))
        def identity():
            stat = p.stat()
            return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        before = identity()
        if self.checked_files.get(p) == before:
            return
        self._inspect(p)
        if identity() != before:
            raise ContinuationError("synthetic input changed during recursive inspection")
        self.checked_files[p] = before

    def _inspect(self, p):
        if p.suffix == ".json":
            value = read_json(p)
            if p.parent == self.root / "monitor_events":
                if (value.get("monitor_installed") is not True or value.get("interpreter") != self.python
                        or value.get("cwd") != str(self.scientific_root)):
                    raise ContinuationError("child monitor environment mismatch")
                for command in value["dispatches"]:
                    self.argv(command)
                return  # sys.path is a diagnostic observation, not a data input.
            self.value(value)
        elif p.suffix == ".jsonl":
            from .identity import canonical
            import json
            for line in p.read_text().splitlines():
                item = strict_json(line)
                canonical(item)  # rejects non-finite values
                self.value(item)
        elif p.suffix in {".yaml", ".yml"}:
            import yaml
            self.value(yaml.safe_load(p.read_text()))
        elif p.suffix == ".pt":
            import torch
            value = torch.load(p, map_location="cpu", weights_only=True)
            if not isinstance(value, dict) or value.get("synthetic_test_only") is not True:
                raise ContinuationError("synthetic checkpoint marker required")
            self.value(value)
        elif p.suffix == ".csv":
            with p.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    self.value(row)

    def tree(self, path=None):
        root = self.path(str(path or self.root))
        for p in root.rglob("*"):
            self.path(str(p))
            if p.is_file():
                self.input_file(p)

    def argv(self, argv):
        if not isinstance(argv, list) or len(argv) < 2 or argv[0] != self.python:
            raise ContinuationError("wrong interpreter or empty child command")
        self.path(argv[1], source=True)
        for index, arg in enumerate(argv[2:], 2):
            if arg == str(self.scientific_root) and argv[index-1] in {"--repository-root", "--clean-worktree-root"}:
                continue
            if "{" in arg or "/ABSOLUTE/" in arg:
                raise ContinuationError("unresolved child argument")
            self.value(arg, "argv")
        return argv
