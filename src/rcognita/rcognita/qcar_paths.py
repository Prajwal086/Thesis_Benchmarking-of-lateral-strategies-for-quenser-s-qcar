"""ROS-independent reference paths for QCar experiments (meters, radians)."""
import numpy as np


def make_path(path_type='circle', points=240, radius=1.0):
    """Return [x, y, heading] reference points, ordered in travel direction.

    Circle and figure_eight are periodic. s_curve and straight are open.
    All start near x=radius,y=0 (except open paths, x=0,y=0).
    """
    if points < 20 or radius <= 0:
        raise ValueError('points >=20 and radius >0 are required')
    if path_type == 'circle':
        u = np.linspace(0, 2*np.pi, points, endpoint=False)
        x, y = radius*np.cos(u), radius*np.sin(u)
        heading = u + np.pi/2
    elif path_type in ('figure_eight', 'infinity'):
        u = np.linspace(0, 2*np.pi, points, endpoint=False)
        x, y = radius*np.sin(u), 0.5*radius*np.sin(2*u)
        dx, dy = radius*np.cos(u), radius*np.cos(2*u)
        heading = np.arctan2(dy, dx)
    elif path_type == 'straight':
        x = np.linspace(0, 4*radius, points)
        y = np.zeros_like(x)
        heading = np.zeros_like(x)
    elif path_type in ('s_curve', 'mild_curve'):
        x = np.linspace(0, 6*radius, points)
        amp = (0.6 if path_type == 's_curve' else 0.2)*radius
        y = amp*np.sin(np.pi*x/(3*radius))
        heading = np.arctan2(np.gradient(y), np.gradient(x))
    else:
        raise ValueError(f'Unknown path_type: {path_type}')
    return np.column_stack((x, y, np.arctan2(np.sin(heading), np.cos(heading))))
