"""Five cell-stage adapters; all writes/publication use the original transaction."""
from pathlib import Path

from . import PHASES
from .identity import ContinuationError, read_json


def cell_layout(phase, coordinates, original, native):
    if phase not in PHASES[:5]:
        raise ContinuationError("not a continuation cell phase")
    flag = "--output_root" if "--output_root" in original else "--output-root"
    if flag not in original:
        raise ContinuationError("transactional command lacks output-root")
    final_root = Path(original[original.index(flag)+1])
    cell_id = native.stable_cell_id(phase, coordinates)
    if phase == "formal_cache_policy":
        final_path = final_root / str(coordinates["capacity_label"])
    elif phase == "formal_controller":
        final_path = final_root / cell_id
    else:
        key = {"formal_ablation": "ablation_setting_id", "formal_support": "support_setting_id",
               "formal_scalability": "scalability_setting_id"}[phase]
        final_path = final_root / str(coordinates[key])

    def build(staging, actual_cell_id):
        staged = list(original)
        if phase == "formal_cache_policy":
            staged[staged.index(flag)+1] = str(staging / "artifact" / "benchmark")
            if "--request-replay-path" not in staged:
                raise ContinuationError("cache-policy transaction lacks request replay path")
            staged[staged.index("--request-replay-path")+1] = str(staging / "artifact" / "request_replay.json")
        elif phase == "formal_controller":
            staged[staged.index(flag)+1] = str(staging / "child_output")
        else:
            child = staging / "child_output"
            staged[staged.index(flag)+1] = str(child)
            staged.extend(["--cell-id", actual_cell_id, "--cell-phase", phase,
                           "--cell-output-descriptor-path", str(child / "cell_child_output.json")])
        return staged

    def resolve(staging, actual_cell_id, _completed):
        if phase == "formal_cache_policy":
            artifact = staging / "artifact"
            benchmark = native.single_child_directory(artifact / "benchmark")
            if not (benchmark / "aggregate_summary.json").is_file():
                raise ContinuationError("cache-policy benchmark aggregate is missing")
            return artifact, ["request_replay.json"], artifact / "benchmark"
        child = staging / "child_output"
        if phase == "formal_controller":
            return native.single_child_directory(child), ["aggregate_summary.json", "benchmark_rows.csv"], child
        artifact, descriptor = native.resolve_child_output_descriptor(
            child / "cell_child_output.json", output_root=child,
            expected_cell_id=actual_cell_id, expected_phase=phase, expected_setting_id=final_path.name)
        if read_json(artifact / "support_provenance.json").get("setting_id") != final_path.name:
            raise ContinuationError("support provenance setting identity mismatch")
        return artifact, descriptor["required_payload"], child

    return final_path, build, resolve
