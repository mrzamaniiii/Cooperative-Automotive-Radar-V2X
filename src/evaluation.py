import numpy as np
from scipy.optimize import linear_sum_assignment

MATCH_GATE = 1.0
WARMUP_TIME = 1.0

def evaluate_multi_tracker(history, ground_truth, times):
    position_errors = []
    false_tracks = []
    missed_targets = []
    matched_counts = []
    for k, t in enumerate(times):
        if t < WARMUP_TIME:
            continue
        frame = history[k]
        gt = ground_truth[:, k, :]
        ids = list(frame.keys())
        positions = np.array([frame[i][:2] for i in ids], dtype=float).reshape(-1, 2)
        n_gt = len(gt)
        n_tracks = len(positions)
        if n_tracks == 0:
            false_tracks.append(0)
            missed_targets.append(n_gt)
            matched_counts.append(0)
            continue
        D = np.linalg.norm(gt[:, None, :] - positions[None, :, :], axis=2)
        penalty = MATCH_GATE + 1.0
        size = n_gt + n_tracks
        C = np.zeros((size, size))
        C[:n_gt, :n_tracks] = np.where(D <= MATCH_GATE, D, 3 * penalty)
        C[:n_gt, n_tracks:] = penalty
        C[n_gt:, :n_tracks] = penalty
        rows, cols = linear_sum_assignment(C)
        valid = [(i, j) for i, j in zip(rows, cols) if i < n_gt and j < n_tracks and (D[i, j] <= MATCH_GATE)]
        for i, j in valid:
            position_errors.append(D[i, j] ** 2)
        matched = len(valid)
        matched_counts.append(matched)
        false_tracks.append(n_tracks - matched)
        missed_targets.append(n_gt - matched)
    rmse = np.sqrt(np.mean(position_errors)) if position_errors else np.nan
    return {'rmse': rmse, 'false_track_frames': sum(false_tracks), 'missed_target_frames': sum(missed_targets), 'mean_matched': np.mean(matched_counts), 'false_timeline': np.array(false_tracks), 'missed_timeline': np.array(missed_targets)}
