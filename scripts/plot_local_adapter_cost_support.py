"""Export a completed local cost audit; never run a model, transfer, or auditor."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_file(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(root, expected_plan, expected_audit, expected_delivery, output):
    inputs = {}
    for name, expected in (("plan.json", expected_plan), ("audit.json", expected_audit),
                           ("delivery.json", expected_delivery)):
        path = root / name
        if digest(path) != expected:
            raise ValueError("input SHA mismatch: " + name)
        inputs[name] = json.loads(path.read_text())
    plan, audit, delivery = (inputs[k] for k in ("plan.json", "audit.json", "delivery.json"))
    if (audit["status"] != "AUDITED_PASS_TECHNICAL_COMPONENT_COST_ONLY"
            or audit["synthetic_only"] or delivery["status"] != "PASS"
            or delivery["audit_sha256"] != expected_audit
            or delivery["plan_sha256"] != expected_plan
            or delivery["audit_invocations"] != 1 or delivery["audit_exit_code"] != 0):
        raise ValueError("only a bound, completed real technical audit can be exported")
    summaries = []
    for key, group in audit["aggregates"].items():
        values = group["all_ms"]
        if (group["n"] != 3 or len(values) != 3 or group["is_CI"]
                or not all(math.isfinite(v) and v >= 0 for v in values)
                or not math.isclose(statistics.mean(values), group["mean_ms"], abs_tol=1e-9)
                or not math.isclose(statistics.stdev(values), group["sample_sd_ms"], abs_tol=1e-9)):
            raise ValueError("invalid three-trial aggregate: " + key)
        summaries.append({"group": key, "n": 3, "trial_1_ms": values[0],
                          "trial_2_ms": values[1], "trial_3_ms": values[2],
                          **{k: group[k] for k in ("mean_ms", "sample_sd_ms", "median_ms", "min_ms", "max_ms")},
                          "is_CI": False})
    intervals = []
    for sample in audit["samples"]:
        for level in ("partitioned_ns", "nested_ns"):
            occurrences = {}
            for span in sample[level]:
                occurrence = occurrences.get(span["kind"], 0)
                occurrences[span["kind"]] = occurrence + 1
                intervals.append({"attempt": sample["id"], "qualification": sample["qualification"],
                                  "status": sample["status"], "level": level,
                                  "kind": span["kind"], "occurrence": occurrence,
                                  "ns": span["ns"], "ms": span["ns"] / 1e6,
                                  "censored_upper_bound": span["censored_upper_bound"]})
        intervals.append({"attempt": sample["id"], "qualification": sample["qualification"],
                          "status": sample["status"], "level": "process_residual",
                          "kind": "unpartitioned_startup_supervision_or_idle", "occurrence": 0,
                          "ns": sample["process_residual_ns"], "ms": sample["process_residual_ns"] / 1e6,
                          "censored_upper_bound": False})
    output.mkdir(parents=True, exist_ok=False)
    csv_file(output / "component_summary.csv", summaries)
    csv_file(output / "native_intervals.csv", intervals)
    bill = [{"item": key, "value": value, "scope": "native operation starts including qualifiers"}
            for key, value in audit["raw_calls"].items()]
    bill += [{"item": key, "value": audit[key], "scope": "automatic auditor"}
             for key in ("utility_process_launch_attempts", "payload_charge_upper_bytes",
                         "application_charge_upper_bytes", "retained_bytes")]
    bill += [{"item": key, "value": delivery[key], "scope": "producer plus automatic terminal consumer"}
             for key in ("combined_seconds", "consumer_seconds", "combined_retained_bytes", "combined_log_bytes")]
    csv_file(output / "resource_bill.csv", bill)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.45))
    colors = {"A": "#D55E00", "B": "#0072B2"}

    def point(axis, position, key, color):
        values = audit["aggregates"][key]
        axis.errorbar(position, values["mean_ms"], yerr=values["sample_sd_ms"],
                      fmt="D", ms=4.2, color=color, capsize=3, lw=1.2, zorder=3)
        axis.scatter([position-.045, position, position+.045], values["all_ms"],
                     s=18, color=color, alpha=.75, edgecolor="white", linewidth=.35, zorder=4)

    labels = ["Processor", "Base", "Adapter", "Activation\n+ checks"]
    for j, kind in enumerate(("processor_load", "base_load", "adapter_load", "activate")):
        for name, shift in (("A", -.16), ("B", .16)):
            adapter = name if kind in ("adapter_load", "activate") else ""
            point(axes[0], j+shift, f"T1|{name}|{kind}|{adapter}|0", colors[name])
    axes[0].set_xticks(range(4), labels)
    axes[0].set_title("(a) Process-first resource preparation", fontsize=10, pad=12)
    axes[0].legend(handles=[Line2D([], [], marker="o", ls="", color=colors["A"], label="ALPR: rank64, DoRA"),
                            Line2D([], [], marker="o", ls="", color=colors["B"], label="Helmet: rank4")],
                   frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylim(10, 2500)
    for j, kind in enumerate(("encode", "durable_save", "restore", "prompt_rebuild")):
        point(axes[1], j, f"T3|state|{kind}||0", "#444444")
    axes[1].set_xticks(range(4), ["Encode", "Durable\nsave", "Restore", "Prompt\nrebuild"])
    axes[1].set_title("(b) Saved application-state components", fontsize=10, pad=12)
    axes[1].set_ylim(.03, 2)
    for j, (obj, color) in enumerate((("state_package", "#444444"), ("adapter_B_weight", colors["B"]),
                                      ("adapter_A_weight", colors["A"]))):
        point(axes[2], j, f"T4|{obj}|transfer||0", color)
    axes[2].set_xticks(range(3), ["State\n2,040 B", "Helmet\n9.64 MB", "ALPR\n154.42 MB"])
    axes[2].set_xlim(-.35, 2.35)
    axes[2].set_title("(c) Instrumented local file transfer", fontsize=10, pad=12)
    axes[2].set_ylim(5, 30000)
    for axis in axes:
        axis.set_yscale("log")
        axis.set_ylabel("Elapsed time (ms, log scale)")
        axis.grid(axis="y", which="major", color="#DDDDDD", lw=.6)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", length=0, pad=8)
    fig.suptitle("Local CPU component measurements", fontsize=13, y=.98)
    fig.text(.5, .085, "Three measured trials per component; dots = every trial, diamonds = mean, whiskers = sample SD (not CI). Qualifiers are separate in CSV.", ha="center", fontsize=8)
    fig.text(.5, .038, "Apple M5 / CPU float32 / one thread. Instrumented components exclude process residuals; local file transfer is not a wireless or pure TCP measurement.", ha="center", fontsize=8)
    fig.subplots_adjust(left=.065, right=.99, bottom=.26, top=.81, wspace=.37)
    for suffix in ("pdf", "svg", "png"):
        path = output / ("cost_components." + suffix)
        with path.open("xb") as handle:
            fig.savefig(handle, format=suffix, dpi=600, facecolor="white")
    plt.close(fig)
    caption = ("Local instrumented component costs from one Apple M5 host, CPU float32, batch1, one intra/inter-op thread. "
               "Every formal trial (ordinals1-3) is shown; diamonds and whiskers denote mean and one sample SD, not CI. "
               "Ordinal0 qualifiers are preserved separately in native_intervals.csv. Process-first does not mean storage-cold; resources were hashed before timing. "
               "Activation includes set_adapter, status and parameter integrity verification, not pure switch latency. "
               "Local transfer includes source/chunk hashing, write-ahead logging, disk guards, receiver file/fsync and ACK/teardown; it is not pure network throughput. MB denotes10^6bytes; exact object bytes are in the bound plan. "
               "Nested intervals must not be added to their parents; process residuals remain separately reported. "
               "All groups, including generation and resident A-B-A, are in component_summary.csv; generation lengths need not be equal, so adapter latencies are not a causal speed comparison. "
               "This technical suffix has no applicable task groundtruth. No wireless/RSU measurement, current-stream parameter calibration or break-even claim is supported. "
               "paper_goal_complete=false.\n")
    with (output / "caption.txt").open("x") as handle:
        handle.write(caption)
    manifest = {"source_commit": plan["source_commit"], "expected_plan_sha256": expected_plan,
                "expected_audit_sha256": expected_audit, "expected_delivery_sha256": expected_delivery,
                "summary_groups": len(summaries), "native_interval_rows": len(intervals),
                "samples": len(audit["samples"]), "new_models_or_transfers_or_audits": 0,
                "paper_goal_complete": False, "files": []}
    manifest["files"] = [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
                         for p in sorted(output.iterdir()) if p.is_file()]
    with (output / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return {k: v for k, v in manifest.items() if k != "files"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execution_root", type=Path, required=True)
    p.add_argument("--expected_plan_sha", required=True)
    p.add_argument("--expected_audit_sha", required=True)
    p.add_argument("--expected_delivery_sha", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(export(args.execution_root, args.expected_plan_sha, args.expected_audit_sha,
                            args.expected_delivery_sha, args.output)))


if __name__ == "__main__":
    main()
