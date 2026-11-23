"""Evaluate the finite-pilot regret bound at all corners of each small star."""
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from pilot_budget_experiment import water_fill

RESULTS = Path(__file__).resolve().parents[1] / "results"


def corner_bound(star, floor):
    fixed = np.asarray(star["omega"])
    if len(fixed) > 8:
        raise ValueError("This exhaustive diagnostic is for the published degree-at-most-eight grids")
    bounds = zip(star["c_lower"], star["c_upper"])
    worst = 0.0
    for corner in itertools.product(*bounds):
        norms = np.asarray(corner)
        optimal = water_fill(norms, floor)
        regret = np.dot(norms**2, 1 / fixed - 1 / optimal)
        worst = max(worst, float(regret))
    return worst


if __name__ == "__main__":
    pilots = json.loads((RESULTS / "pilot_budget_weights.json").read_text())
    records = []
    for pilot in pilots:
        bound = sum(corner_bound(star, pilot["floor_fraction"]) for star in pilot["stars"])
        separate = pilot["regret_bound"]
        if bound > separate + 1e-10 * max(1, separate):
            raise ArithmeticError("Corner bound exceeds the separate-envelope bound")
        records.append(dict(case=pilot["key"], separate=separate, corner=bound,
                            ratio=bound/separate, corner_over_reference_P=bound/pilot["P_floor_oracle_reference"]))
    with (RESULTS / "pilot_box.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    print(f"Completed {len(records)} finite-pilot box diagnostics")
