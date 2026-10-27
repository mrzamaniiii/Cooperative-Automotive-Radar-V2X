import numpy as np

def is_occluded(sensor, target, obstacle, radius):
    direction = target - sensor
    denominator = np.dot(direction, direction)
    if denominator < 1e-12:
        return False
    u = np.clip(np.dot(obstacle - sensor, direction) / denominator, 0.0, 1.0)
    closest = sensor + u * direction
    distance = np.linalg.norm(closest - obstacle)
    return distance <= radius
