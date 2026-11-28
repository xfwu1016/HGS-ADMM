"""Validate and merge the declared 14 TCP arms without discarding duplicates."""
import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_records(rows, require_full=True):
    keys = [(r['dataset'], int(r['seed']), r['profile'], r['weight']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate TCP run key; retain repetitions under distinct IDs')
    expected = {(ds, seed, profile, weight)
                for ds, seed, profile in
                [('diabetes', s, 'lan') for s in range(2026, 2031)] +
                [('diabetes', 2026, 'wan'), ('breast_cancer', 2026, 'lan')]
                for weight in ('unit', 'budget')}
    if require_full and set(keys) != expected:
        raise ValueError(f'Unexpected TCP design: missing={expected-set(keys)}, extra={set(keys)-expected}')
    for r in rows:
        if str(r['converged']).lower() != 'true':
            raise ValueError('TCP solve did not converge')
        gap = float(r['relative_gap'])
        if not math.isfinite(gap) or not 0 <= gap <= 1e-7:
            raise ValueError('Invalid TCP gap')
        if int(r['iterations']) != int(r['reference_iterations']):
            raise ValueError('TCP/reference stopping iteration mismatch')
        distance = float(r['coefficient_distance_to_reference'])
        difference = float(r['objective_difference_to_reference'])
        if not math.isfinite(distance) or not math.isfinite(difference) or distance > 1e-8 or abs(difference) > 1e-9:
            raise ValueError('TCP/reference numerical mismatch')
        p = {'diabetes': 65, 'breast_cancer': 30}[r['dataset']]
        k = int(r['iterations'])
        if int(r['payload_bytes']) != k*(72*p+16) or int(r['coordinator_bytes']) != int(r['payload_bytes']):
            raise ValueError('TCP application byte ledger mismatch')
        if int(r['messages']) != 2*k or int(r['coordinator_messages']) != 2*k:
            raise ValueError('TCP application message ledger mismatch')
        if int(r['worker_pid']) == int(r['coordinator_pid']):
            raise ValueError('Worker and coordinator PIDs coincide')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir', type=Path, default=ROOT/'results')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    rows = []
    for name in ['tcp_diabetes_lan_repeated.csv', 'tcp_diabetes.csv', 'tcp_breast_cancer.csv']:
        with (args.results_dir/name).open(newline='') as stream:
            rows.extend(csv.DictReader(stream))
    validate_records(rows)
    rows.sort(key=lambda r: (r['dataset'], r['profile'], int(r['seed']), r['weight']))
    output = args.output or args.results_dir/'tcp_distributed_solver.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {output} ({len(rows)} validated records)')


if __name__ == '__main__':
    main()
