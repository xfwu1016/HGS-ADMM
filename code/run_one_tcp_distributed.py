"""Run one paired-arm component of the two-process TCP experiment."""
from __future__ import annotations
import argparse
import json
from tcp_distributed_solver import one_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--weight', required=True)
    parser.add_argument('--latency', type=float, required=True)
    parser.add_argument('--bandwidth', type=float, required=True)
    args = parser.parse_args()
    result = one_run(
        args.dataset, args.seed, args.weight, args.latency, args.bandwidth,
        tol=1e-7, maxiter=10000,
    )
    print(json.dumps(result))


if __name__ == '__main__':
    main()
