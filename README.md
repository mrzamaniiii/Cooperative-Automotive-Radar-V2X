# Cooperative Automotive Radar & V2X Perception

**Python research prototype** for cooperative multi-object tracking under geometric occlusion, simulated roadside LiDAR messages, and nonlinear radar measurements.

## Scope
- 2D urban simulation with **three** synthetic moving targets and a simplified obstacle.
- Target-level automotive radar observations: **range, azimuth, radial velocity** (not raw 77 GHz FMCW waveforms).
- Noisy 2D roadside LiDAR detections transmitted with synthetic **300 ms delay and 10% independent packet loss**.
- Multi-object extended Kalman filtering with **Mahalanobis gating + Hungarian association**.
- Comparison of **Radar Only**, **Naive Fusion** (process by arrival time), and **Latency-Aware Fusion** (replay received measurements in measurement-time order).

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q
python run_experiment.py
```

Figures and metrics are written to `results/`. Open `notebooks/colab_demo.ipynb` in Google Colab to run using a ZIP of this repository. `notebooks/experimental_history.ipynb` preserves the full development notebook, including exploratory experiments and superseded methods; it is **not** the recommended entry point.

## Single-seed results

The reference run uses seed **2028**, a 12-second scenario at 10 Hz, and evaluates position RMSE only over **successfully matched** tracks from 1 s onward (1 m matching gate). This makes RMSE conditional on successful association; always report unmatched and missed target-frames alongside it. Run `python run_experiment.py` to generate exact results. These are synthetic single-scenario results, **not** deployment or real-world safety claims.

## Limitations

- Current latency-aware method replays the entire history of *arrived* measurements at every evaluation time; it is **causal but not real-time optimized** and it does **not** preserve persistent track identities between replays.
- No track confirmation/deletion in the v2.2 comparison. Later persistent-tracker code in the history notebook is exploratory and is not integrated here.
- No raw FMCW signal chain, CFAR, 5G/C-V2X protocol stack, real sensor calibration, actual vehicle kinematics, or clutter model.
- A fixed **0.8 m birth exclusion** is a scenario-specific heuristic that can hide real close targets.
- The scenario simulator uses ground-truth target labels internally to generate observations, but the tracker is not supplied with those labels.
- No claim of affiliation with or reuse of IoTLab's implementation. Research ideas should be cited when directly relied upon.

## Repository map

- `src/scenario.py`: synthetic observations and delivery timestamps
- `src/geometry.py`: geometric line-of-sight occlusion
- `src/tracking.py`: EKF and multi-sensor association, prototype replay
- `src/evaluation.py`: matching-based evaluation
- `tests/`: smoke and determinism tests
- `notebooks/`: Colab demo and historical development notebook

## Reproducibility and provenance

This repository reorganizes the Python code developed in the uploaded Colab notebook `Untitled3.ipynb`. NumPy, SciPy and Matplotlib provide the numerical building blocks. The implementation is a self-contained synthetic experiment; no source code from an IoTLab repository was provided or intentionally imported. Standard EKF/Hungarian methods are established public algorithms, and application-specific literature should be cited appropriately when discussing related work.

## Next steps

Persistent track lifecycle, rigorous ID association metrics, richer near-target tests, Monte Carlo analysis of the *integrated* three-target tracker, asynchronous out-of-sequence filtering with bounded replay, and eventually FMCW signal-level front-end integration.
