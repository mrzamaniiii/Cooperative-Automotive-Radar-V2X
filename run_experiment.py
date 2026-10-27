"""Reproduce v2.2 research-prototype results from the uploaded Colab workflow."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.scenario import generate
from src.tracking import tracker_at
from src.evaluation import evaluate_multi_tracker

def main():
    times, truth, events = generate(seed=2028)
    radar = [e for e in events if e.sensor == 'radar']
    methods = {'Radar Only': (radar, True), 'Naive Fusion': (events, False),
               'Latency-Aware Fusion': (events, True)}
    outputs = {name: [tracker_at(float(t), stream, aware) for t in times]
               for name, (stream, aware) in methods.items()}
    metrics = {name: evaluate_multi_tracker(history, truth, times)
               for name, history in outputs.items()}
    for name, m in metrics.items():
        print(f"{name:22s} RMSE={m['rmse']:.4f}m  unmatched={m['false_track_frames']}  missed={m['missed_target_frames']}")
    dest = Path('results'); dest.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for j in range(3):
        ax.plot(truth[j, :, 0], truth[j, :, 1], label=f'Truth {j}')
    for tid in sorted({tid for h in outputs['Latency-Aware Fusion'] for tid in h}):
        xy = np.array([h.get(tid, np.full(4, np.nan))[:2] for h in outputs['Latency-Aware Fusion']])
        ax.plot(xy[:, 0], xy[:, 1], '--', label=f'EKF {tid}')
    ax.set(xlabel='X (m)', ylabel='Y (m)', title='Cooperative tracking — v2.2 prototype')
    ax.grid(True); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(dest/'tracks.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(list(metrics), [m['rmse'] for m in metrics.values()])
    ax.set(ylabel='Matched-target position RMSE (m)', title='Single-seed comparison')
    ax.tick_params(axis='x', labelrotation=12)
    fig.tight_layout(); fig.savefig(dest/'rmse.png', dpi=160); plt.close(fig)
    (dest/'metrics.json').write_text(json.dumps({name: {k: float(v) for k, v in m.items() if np.isscalar(v)} for name,m in metrics.items()},indent=2))

if __name__ == '__main__':
    main()
