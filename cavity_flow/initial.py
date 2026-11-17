import numpy as np

# Generating null scalar fields (cavity is still at the beginning)
def apply_initial_conditions(grid):

    u = np.zeros((grid.Ny, grid.Nx))
    v = np.zeros((grid.Ny, grid.Nx))
    p = np.zeros((grid.Ny, grid.Nx))    # kinematic pressure, defined up to a constant (only its gradient enters the equations)

    return u, v, p
