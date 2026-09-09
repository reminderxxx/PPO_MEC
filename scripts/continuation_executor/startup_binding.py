"""Source-controlled bindings for the public startup host.

Only ``ProductionStartupBinding`` is reachable from the production CLI.  The
synthetic binding is constructed by the repository acceptance entry and rejects
every production root/domain.
"""
from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
import uuid

from .identity import ContinuationError, canonical, within


class ProductionStartupBinding:
    domain = "production"

    def after_startup_event(self, row):
        del row

    def context(self, contract):
        if contract["domain"] != "production" or contract["fixture_root"] is not None:
            raise ContinuationError("production entry refuses synthetic trust")
        from .production_trust import new_production_context
        return new_production_context()

    def verify(self, contract, approval, context, now):
        from .authorization import verify_approval
        return verify_approval(contract, approval, now=now,
                               _production_context=context)

    def qualify(self, proposal, contract):
        from .scientific import QualifiedRun
        return QualifiedRun(proposal, contract)

    def qualification_report(self, qualified):
        return {"reconciliation": qualified.reconciliation}

    def execute(self, qualified, phase, authorize, identity, *, finalize_only=False):
        from .execution import execute_phase
        return execute_phase(qualified, phase, authorize, identity,
                             finalize_only=finalize_only)


class SyntheticAcceptanceBinding:
    """Narrow public-flow adapter: test root only, no scientific capability."""
    domain = "synthetic"

    def __init__(self, installation):
        self.pin = installation
        self.fixture = Path(installation["fixture_root"])
        self._context = None
        if installation.get("domain") != "synthetic" or not self.fixture.name.startswith("synthetic_"):
            raise ContinuationError("synthetic startup binding requires a synthetic_* root")

    def context(self, contract):
        if (contract["domain"] != "synthetic"
                or contract["fixture_root"] != str(self.fixture)
                or not Path(contract["run_root"]).name.startswith("synthetic_")):
            raise ContinuationError("synthetic binding scope mismatch")
        from .production_trust import TrustContext
        self._context = TrustContext(self.pin, test_only=True)
        return self._context

    def after_startup_event(self, row):
        """Optional FIFO barrier owned solely by the synthetic acceptance driver."""
        if row.get("event") != "challenge":
            return
        barrier = self.fixture / "challenge_release_test_only.fifo"
        try:
            mode = os.stat(barrier, follow_symlinks=False).st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISFIFO(mode):
            raise ContinuationError("synthetic challenge barrier is not a FIFO")
        with barrier.open("rb", buffering=0) as stream:
            if stream.read(1) != b"1":
                raise ContinuationError("synthetic challenge barrier was not released")

    def verify(self, contract, approval, context, now):
        from .authorization import verify_approval
        if context is not self._context:
            raise ContinuationError("synthetic context identity changed")
        return verify_approval(contract, approval, test_trust_context=context, now=now)

    def qualify(self, proposal, contract):
        del proposal
        if contract["domain"] != "synthetic":
            raise ContinuationError("synthetic qualification domain")
        imported = sorted(name for name in sys.modules if name.startswith("src.")
                          or name == "continuation_executor.scientific")
        if imported:
            raise ContinuationError("science imported before synthetic admission")
        return {"fixture_root": str(self.fixture), "run_root": contract["run_root"],
                "mode": "synthetic_public_startup_acceptance"}

    def qualification_report(self, qualified):
        return {"synthetic_qualification": qualified,
                "scientific_import_count": 0,
                "original_run_writer_lock_touch_count": 0,
                "original_run_write_count": 0}

    def execute(self, qualified, phase, authorize, identity, *, finalize_only=False):
        del identity, finalize_only
        grant = authorize()
        if not grant.get("approval_verified") or grant.get("domain") != "synthetic":
            raise ContinuationError("synthetic execute requires verified grant")
        root = within(qualified["fixture_root"], qualified["fixture_root"])
        output = root / "synthetic_dispatches"
        output.mkdir(exist_ok=True)
        marker = output / (phase + "-" + uuid.uuid4().hex + ".json")
        payload = {"version": "1.0.0", "test_only": True, "phase": phase,
                   "context_request": self._context.startup_request(),
                   "real_execution_authorized": False,
                   "scientific_rollout_count": 0, "real_v16_dispatch_count": 0,
                   "real_v16_write_count": 0, "holdout_consumption_count": 0,
                   "synthetic_dispatch_count": 1}
        with marker.open("xb") as stream:
            stream.write(canonical(payload) + b"\n")
            stream.flush()
        return {**payload, "marker": str(marker)}
