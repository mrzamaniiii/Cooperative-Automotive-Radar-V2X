import numpy as np
from itertools import groupby
from scipy.optimize import linear_sum_assignment
from .scenario import RADAR_STD, LIDAR_STD

class MultiEKF:

    def __init__(self, z, sensor, t, track_id):
        self.id = track_id
        self.t = float(t)
        self.hits = 1
        if sensor == 'radar':
            r, angle, vr = z
            c, s = (np.cos(angle), np.sin(angle))
            self.x = np.array([r * c, r * s, vr * c, vr * s])
            self.P = np.diag([0.5, 0.5, 4.0, 4.0])
        else:
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            self.P = np.diag([0.12 ** 2, 0.12 ** 2, 4.0, 4.0])

    def predict(self, t):
        dt = float(t - self.t)
        if dt < -1e-08:
            raise ValueError('Backward prediction')
        dt = max(dt, 0.0)
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        Q = 0.6 ** 2 * G @ G.T
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = float(t)

    def model(self, sensor):
        if sensor == 'lidar':
            H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            zhat = H @ self.x
            R = np.eye(2) * LIDAR_STD ** 2
        else:
            px, py, vx, vy = self.x
            r = max(np.hypot(px, py), 1e-06)
            r2 = r * r
            dot = px * vx + py * vy
            zhat = np.array([r, np.arctan2(py, px), dot / r])
            H = np.array([[px / r, py / r, 0, 0], [-py / r2, px / r2, 0, 0], [vx / r - dot * px / r ** 3, vy / r - dot * py / r ** 3, px / r, py / r]])
            R = np.diag(RADAR_STD ** 2)
        return (zhat, H, R)

    def innovation(self, z, sensor):
        zhat, H, R = self.model(sensor)
        y = np.asarray(z) - zhat
        if sensor == 'radar':
            y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.P @ H.T + R
        d2 = float(y @ np.linalg.solve(S, y))
        return (y, H, R, S, d2)

    def update(self, z, sensor):
        y, H, R, S, _ = self.innovation(z, sensor)
        K = np.linalg.solve(S, H @ self.P).T
        self.x += K @ y
        A = np.eye(4) - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        self.P = (self.P + self.P.T) / 2
        self.hits += 1

def associate_scan(tracks, detections, sensor, t, next_id):
    for tr in tracks:
        tr.predict(t)
    if not detections:
        return next_id
    n, m = (len(tracks), len(detections))
    threshold = 11.34 if sensor == 'radar' else 9.21
    cost = np.full((n, m), 1000000.0)
    for i, tr in enumerate(tracks):
        for j, det in enumerate(detections):
            d2 = tr.innovation(det.z, sensor)[4]
            if d2 <= threshold:
                cost[i, j] = d2
    assigned = set()
    if n and m:
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if cost[i, j] <= threshold:
                tracks[i].update(detections[j].z, sensor)
                assigned.add(j)
    for j, det in enumerate(detections):
        if j in assigned:
            continue
        if sensor == 'radar':
            r, angle = det.z[:2]
            pos = np.array([r * np.cos(angle), r * np.sin(angle)])
        else:
            pos = np.asarray(det.z[:2])
        if any((np.linalg.norm(tr.x[:2] - pos) < 0.8 for tr in tracks)):
            continue
        tracks.append(MultiEKF(det.z, sensor, t, next_id))
        next_id += 1
    return next_id

def tracker_at(current_time, all_events, aware=True):
    available = [e for e in all_events if e.arrival_time <= current_time + 1e-08]
    available.sort(key=lambda e: (e.measurement_time if aware else e.arrival_time, 0 if e.sensor == 'radar' else 1))
    tracks = []
    next_id = 0
    for event_t, group in groupby(available, key=lambda e: e.measurement_time if aware else e.arrival_time):
        batch = list(group)
        for sensor in ['radar', 'lidar']:
            scan = [e for e in batch if e.sensor == sensor]
            if scan:
                next_id = associate_scan(tracks, scan, sensor, event_t, next_id)
    for tr in tracks:
        tr.predict(current_time)
    return {tr.id: tr.x.copy() for tr in tracks}
