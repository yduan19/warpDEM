import numpy as np

def _particle_grid(dim_x, dim_y, dim_z, lower, radius, jitter = 0.1):
    points = np.meshgrid(np.linspace(0, dim_x, dim_x), np.linspace(0, dim_y, dim_y), np.linspace(0, dim_z, dim_z))
    points_t = np.array((points[0], points[1], points[2])).T * radius * 4.8 + np.array(lower)
    points_t = points_t + np.random.rand(*points_t.shape) * radius * jitter

    points_t = points_t.reshape((-1, 3))
    points_t[:,1] = points_t[:,1] + 20
    return points_t