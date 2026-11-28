# Reproducible environment

The environment snapshot uses Python **3.12.14** and these main packages:

| Package | Version |
|---|---:|
| NumPy | 2.3.5 |
| SciPy | 1.17.0 |
| scikit-learn | 1.8.0 |
| Matplotlib | 3.10.8 |
| pandas | 2.2.3 |

The recorded fresh run used Linux x86-64 with an AMD EPYC 9V74 processor, nine CPUs visible in the process affinity mask, three independent jobs where supported, and one BLAS thread per solver. The exact recorded platform string and validation results are in [RUN_VERIFICATION.json](../RUN_VERIFICATION.json).

[requirements.txt](../requirements.txt) pins those five packages. [requirements-lock.txt](../requirements-lock.txt) also records their installed runtime dependencies, including `threadpoolctl`, which is directly used in benchmark control. The snapshot was obtained by following installed package metadata with optional extras disabled. It records package versions, not operating-system libraries, wheel hashes or BLAS binaries.

Use a fresh Python 3.12 virtual environment and install the lock file, as shown in the [README](../README.md). `reproduce.py` checks that the main dependencies are present and reports version differences; it does not reject every version mismatch. For a reproduction with matching package versions, use the lock file rather than treating the presence check as a strict environment certification.

## CPU, processes and plotting

The launcher sets `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` for its child scripts. The `--jobs` value controls parallel independent work in selected stages, not the number of BLAS threads inside an optimization fit. No GPU or cluster service is used.

The two TCP roles use Python `multiprocessing` with the `spawn` method and loopback address `127.0.0.1`. The system must permit child processes and local sockets. Their modeled delays are actual waits, so the process-level study has a nonzero timing cost even on a fast machine.

Matplotlib uses a noninteractive backend. Available fonts can differ across operating systems; the scripts request Arial, Helvetica or Liberation Sans and Matplotlib can fall back when needed. PDF/SVG/PNG rendering and file bytes therefore need not match across platforms. No manuscript typesetting software is required for numerical reproduction.

Each execution writes a JSON manifest under `logs/` with Python/platform information, main package versions, stage arguments, elapsed times, return codes and script hashes. The centralized benchmark also writes `results/centralized_benchmark_environment.json` with the CPU and numerical-library/thread information for that run. Those records describe the executed environment; they should be kept with any reported timing results.

## Offline installation

After dependencies are installed, the full experiment uses only local files and loopback TCP. For a machine without internet access, prepare wheels on a connected machine with the **same operating system, CPU architecture and Python version**:

```bash
python3.12 -m pip download --only-binary=:all: -r requirements-lock.txt -d wheelhouse
```

Transfer this code directory and the `wheelhouse` directory to the offline machine, create its virtual environment, and install locally:

```bash
.venv/bin/python -m pip install --no-index --find-links wheelhouse -r requirements-lock.txt
.venv/bin/python reproduce.py --validate
.venv/bin/python reproduce.py --all --jobs 3
```

On Windows use `.venv\Scripts\python.exe`. The GitHub code archive does not include Python, wheels or an operating-system image. If a compatible wheel is unavailable for the target platform, the wheel-download command fails explicitly; prepare a supported matching environment before attempting the offline run.

## Runtime and storage

The validation suite and the full study have very different scopes. Full execution runs 38 stages and many optimization fits, including pilot grids, continuation controls, strict stopping targets and the delayed TCP study. The recorded fresh Linux run completed in 888.52 seconds (about 14.8 minutes), using `--jobs 3` and one BLAS thread per solver. This measurement describes that run rather than a general performance guarantee. Exact completion times are hardware-dependent and are recorded in the execution manifest. A `--jobs 1` run reduces simultaneous resource use; the numerical configurations remain the same.

Generated outputs and logs are ignored by `.gitignore` so repeated local runs do not silently become part of a public commit. Keep a separate copy of completed output directories when preserving a particular run. No destructive cleanup command is required by the documented workflow.
