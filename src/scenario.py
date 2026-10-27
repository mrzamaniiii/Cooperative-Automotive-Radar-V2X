import numpy as np
from dataclasses import dataclass
from .geometry import is_occluded

@dataclass
class Detection:
    arrival_time: float
    measurement_time: float
    sensor: str
    z: np.ndarray

RADAR_STD = np.array([0.20, np.deg2rad(0.5), 0.10])
LIDAR_STD = 0.12

def generate(seed=2028, latency=0.3, packet_loss=0.10):
    rng = np.random.default_rng(seed)
    dt = 0.1
    times = np.round(np.arange(0, 12.01, dt), 6)
    ego, roadside = np.array([0., 0.]), np.array([22., 15.])
    truth = np.stack([
        np.column_stack((10 + 1.2*times, 2 + 0.15*times)),
        np.column_stack((31 - 0.9*times, 4 - 0.10*times)),
        np.column_stack((24 + 0.8*times, -1 + 0.5*times)),
    ])
    velocities = np.array([[1.2, 0.15], [-0.9, -0.10], [0.8, 0.5]])
    events = []
    for k, t in enumerate(times):
        blocker = np.array([12., 0.]) if t < 5 else np.array([12., 8.])
        for obj in range(3):
            p, v = truth[obj, k], velocities[obj]
            if not is_occluded(ego, p, blocker, 2.):
                relative = p - ego
                r = np.linalg.norm(relative)
                z = np.array([r, np.arctan2(relative[1], relative[0]), np.dot(relative, v)/r])
                z += rng.normal(0, RADAR_STD)
                z[1] = (z[1] + np.pi) % (2*np.pi) - np.pi
                events.append(Detection(float(t), float(t), 'radar', z))
            if not is_occluded(roadside, p, blocker, 2.) and rng.random() >= packet_loss:
                z = p + rng.normal(0, LIDAR_STD, 2)
                arrival = np.ceil((t + latency - 1e-9) / dt)*dt
                events.append(Detection(round(float(arrival), 6), float(t), 'lidar', z))
    events.sort(key=lambda e: e.arrival_time)
    return times, truth, events
