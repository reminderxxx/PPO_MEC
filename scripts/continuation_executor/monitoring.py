"""Observed call/write/dispatch evidence for isolated acceptance processes."""
from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import sys

from .identity import ContinuationError

PROTECTED_V16 = "/Users/howen/Projects/PPO_MEC/artifacts/experiments/typed_model_cache_formal/typed_model_cache_formal_20260906_152847_g14c_v16"


class Monitor:
    """Install before scientific imports; an observer is not a validator mock.

    The audit hook cannot be removed. Use one monitor per dedicated process.
    Reports distinguish observed attempted violations from completed execution.
    """
    def __init__(self, fixture_root=None, *, real_run_root=PROTECTED_V16):
        self.fixture = Path(fixture_root).resolve() if fixture_root else None
        self.real_run = Path(real_run_root).resolve() if real_run_root else None
        self.calls = Counter()
        self.dispatches = []
        self.write_events = 0
        self.scientific_rollout_count = 0
        self.real_v16_dispatch_count = 0
        self.real_v16_write_count = 0
        self.denied = []
        self.previous_profile = sys.getprofile()
        sys.setprofile(self.profile)
        sys.addaudithook(self.audit)

    def profile(self, frame, event, arg):
        if event != "call":
            return
        module, name = frame.f_globals.get("__name__", "") or "", frame.f_code.co_name
        if module == "__main__" and "/scripts/" in frame.f_code.co_filename:
            module = "scripts." + frame.f_code.co_filename.rsplit("/scripts/", 1)[1].removesuffix(".py").replace("/", ".")
        if module.startswith(("src.", "scripts.")):
            self.calls[module + "." + name] += 1
            forbidden = (module.startswith("src.envs.") and name in {"step", "reset"}) or (
                "rollout" in name or name in {"run_real_episode", "train", "learn"})
            if forbidden:
                self.scientific_rollout_count += 1
                self.denied.append({"kind":"scientific_execution", "symbol":module+"."+name})
                raise ContinuationError("acceptance must not execute scientific rollout: " + module + "." + name)

    def audit(self, event, args):
        if event == "subprocess.Popen":
            argv = args[1]
            if isinstance(argv, (list,tuple)):
                argv = [str(value) for value in argv]
            else:
                argv = [str(argv)]
            scientific = any(value.endswith(".py") for value in argv)
            if scientific:
                self.dispatches.append(argv)
                if self.real_run and any(str(self.real_run) in value for value in argv):
                    self.real_v16_dispatch_count += 1
                    raise ContinuationError("real v16 child dispatch forbidden during acceptance")
        paths = []
        if event == "open":
            path, mode, flags = args
            if self.fixture and isinstance(path, (str,bytes,os.PathLike)):
                observed = Path(os.fsdecode(path)).absolute()
                outside = observed != self.fixture and self.fixture not in observed.parents
                if outside and (observed.suffix in {".pt", ".pth", ".csv", ".npz"}
                        or "/data/raw/" in str(observed) or "/artifacts/" in str(observed)):
                    self.denied.append({"kind":"external_data_read", "path":str(observed)})
                    raise ContinuationError("synthetic process attempted external data/weight access")
            if isinstance(path, (str,bytes,os.PathLike)) and (
                    flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):
                paths = [path]
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.truncate", "os.utime"}:
            paths = [args[0]]
        elif event in {"os.rename", "os.link", "os.symlink"}:
            paths = list(args[:2])
        for value in paths:
            if isinstance(value, int):
                self.write_events += 1  # Descriptor was checked when opened.
                continue
            path = Path(os.fsdecode(value)).absolute()
            if str(path) == os.devnull:
                continue  # OS sink, not a persistent artifact write.
            if self.real_run and (path == self.real_run or self.real_run in path.parents):
                self.real_v16_write_count += 1
                raise ContinuationError("real v16 write forbidden during acceptance")
            if self.fixture and path != self.fixture and self.fixture not in path.parents:
                self.denied.append({"kind":"write_escape", "path":str(path)})
                raise ContinuationError("acceptance write outside fixture: " + str(path))
            self.write_events += 1

    def report(self):
        return {"synthetic_child_dispatch_count":len(self.dispatches),
                "scientific_rollout_count":self.scientific_rollout_count,
                "real_v16_dispatch_count":self.real_v16_dispatch_count,
                "real_v16_write_count":self.real_v16_write_count,
                "observed_write_events":self.write_events, "actual_function_calls":dict(self.calls),
                "dispatches":self.dispatches,"denied":self.denied,
                "monitor_installed":True}
