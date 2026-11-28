"""Compare a main or pilot rerun to supplied records, ignoring machine timings."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check_records(original, fresh, where):
    if len(original) != len(fresh):
        raise ValueError(f"{where}: different record counts")
    for index, (left, right) in enumerate(zip(original, fresh)):
        if left.keys() != right.keys():
            raise ValueError(f"{where}: fields differ at row {index}")
        for field in left:
            if "seconds" in field or left[field] == right[field]:
                continue
            try:
                equal = float(left[field]) == float(right[field])
            except (ValueError, TypeError):
                equal = False
            if not equal:
                raise ValueError(f"{where}, row {index}, {field}: {left[field]!r} vs {right[field]!r}")


def check_traces(left, right):
    def read(path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return {row["key"]: row["history"] for row in map(json.loads, f)}
    first, second = read(left), read(right)
    if first.keys() != second.keys():
        raise ValueError(f"{left.name}: trace keys differ")
    for key in first:
        check_records(first[key], second[key], key)
    return {"trajectories": len(first), "checkpoints": sum(map(len, first.values()))}


def check_npz(left, right):
    with np.load(left) as original, np.load(right) as fresh:
        if set(original.files) != set(fresh.files):
            raise ValueError(f"{left.name}: different array keys")
        for key in original.files:
            if not np.array_equal(original[key], fresh[key]):
                raise ValueError(f"{left.name}/{key}: array mismatch")
        return len(original.files)


def compare(reference, replay, stage):
    report = {"stage": stage, "csv_rows_checked": {}, "comparison": "Exact equality of non-timing values; decimal formatting differences are ignored."}
    names = (["runs.csv", "stopping_certificates.csv", "summary.csv"] if stage == "main"
             else ["pilot_budget_runs.csv", "pilot_budget_pilots.csv", "pilot_budget_summary.csv"])
    for name in names:
        a, b = csv_rows(reference / name), csv_rows(replay / name)
        check_records(a, b, name)
        report["csv_rows_checked"][name] = len(a)
    trace_name = "traces.jsonl.gz" if stage == "main" else "pilot_budget_traces.jsonl.gz"
    report.update(check_traces(reference / trace_name, replay / trace_name))
    if stage == "main":
        paths = sorted(path.relative_to(reference) for folder in ["coefficients", "datasets"]
                       for path in (reference / folder).glob("*.npz"))
    else:
        paths = [Path("pilot_budget_coefficients.npz")]
    report["npz_archives"] = len(paths)
    report["arrays_checked"] = sum(check_npz(reference / path, replay / path) for path in paths)
    report["status"] = "passed"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["main", "pilot"])
    parser.add_argument("reference_results", type=Path)
    parser.add_argument("replay_results", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.reference_results, args.replay_results, args.stage), indent=2))
