"""Three paired, order-randomized timing repeats on the seed-2026 Diabetes LAN split.

This is a within-split timing-variability diagnostic, not an additional
independent split. Each arm runs in a fresh parent Python process to isolate
multiprocessing resource trackers. The canonical 14-run TCP experiment is not
modified. Worker-loop timing excludes process startup and data preprocessing.
"""
from __future__ import annotations
import argparse, csv, json, os, random, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = Path(__file__).resolve().parent

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--order-seed', type=int, default=20261127)
    ap.add_argument('--output', type=Path, default=ROOT/'results'/'tcp_timing_repeats_focus.csv')
    args = ap.parse_args()
    if args.repeats != 3:
        ap.error('The manuscript reports exactly three paired timing repetitions.')
    rng = random.Random(args.order_seed)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    runner = CODE/'run_one_tcp_distributed.py'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.output.open('w', newline='') as stream:
        writer = None
        for rep in range(1, 4):
            order = ['unit', 'budget']; rng.shuffle(order)
            for order_index, weight in enumerate(order, 1):
                cmd = [sys.executable, str(runner), '--dataset', 'diabetes', '--seed', '2026',
                       '--weight', weight, '--latency', '0.1', '--bandwidth', '1000.0']
                proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True,
                                      capture_output=True, timeout=180)
                if proc.returncode:
                    raise RuntimeError(f'Arm failed (rep={rep}, weight={weight}):\n{proc.stdout}\n{proc.stderr}')
                payload = next((line for line in reversed(proc.stdout.splitlines()) if line.startswith('{')), None)
                if payload is None: raise RuntimeError(f'No JSON result: {proc.stdout}')
                row = json.loads(payload)
                row.update(repetition=rep, pair_order=order_index,
                           randomized_order_seed=args.order_seed, profile='lan')
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader()
                writer.writerow(row); stream.flush()
                print(f"rep={rep} order={order_index} {weight}: {row['worker_wall_s']:.6f}s, "
                      f"{row['iterations']} iterations", flush=True)
    print(f'Wrote {len(rows)} validated records to {args.output}')

if __name__ == '__main__': main()
