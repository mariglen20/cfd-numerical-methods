import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu


class PoissonSolver:
    def __init__(self, grid):
        self.grid = grid
        self.A = None
        self.lu = None
        self._build_matrix()

    def _build_matrix(self):
        g = self.grid
        N = g.Nx * g.Ny
        A = lil_matrix((N, N))

        for j in range(g.Ny):
            for i in range(g.Nx):
                k = i + j * g.Nx

                # Pure Neumann problem: p is defined up to a constant,
                # so p is fixed in one boundary node to make A non-singular
                if k == 1:
                    A[k, k] = 1

                # Boundary nodes: dp/dn = 0 (first order one-sided difference)
                elif i == 0 or i == g.Nx-1 or j == 0 or j == g.Ny-1:
                    A[k, k] = 1
                    if i == 0:            A[k, k + 1]    = -1
                    elif i == g.Nx - 1:   A[k, k - 1]    = -1
                    elif j == 0:          A[k, k + g.Nx] = -1
                    elif j == g.Ny - 1:   A[k, k - g.Nx] = -1

                # Interior nodes: 5-point Laplacian
                else:
                    A[k, k]        = -(2/g.dx**2 + 2/g.dy**2)
                    A[k, k - 1]    =  1/g.dx**2
                    A[k, k + 1]    =  1/g.dx**2
                    A[k, k - g.Nx] =  1/g.dy**2
                    A[k, k + g.Nx] =  1/g.dy**2

        self.A = A.tocsc()          # splu requires CSC format
        self.lu = splu(self.A)      # LU factorization computed only once

    def solve(self, rhs):
        g = self.grid
        b = rhs.ravel().copy()
        b[1] = 0                    # reference node, consistent with the matrix row k = 1
        p_vec = self.lu.solve(b)    # only forward/back substitution, no refactorization
        p = p_vec.reshape((g.Ny, g.Nx))
        p -= p.mean()              # zero-mean pressure (only grad(p) matters)
        return p

    def test(self, operators, grid):
        """Solves lap(p) = 1 and prints the algebraic and discrete residuals."""
        rhs = np.zeros((grid.Ny, grid.Nx))
        rhs[1:-1, 1:-1] = 1.0

        b = rhs.ravel().copy()
        b[1] = 0

        p_vec = self.lu.solve(b)

        print(f"p_vec has NaN: {np.any(np.isnan(p_vec))}")
        print(f"p_vec has Inf: {np.any(np.isinf(p_vec))}")
        print(f"p_vec range: [{p_vec.min():.4e}, {p_vec.max():.4e}]")

        alg_res = np.max(np.abs(self.A.dot(p_vec) - b))
        print(f"Algebraic residual max|A p - b|: {alg_res:.4e}")

        p = p_vec.reshape((grid.Ny, grid.Nx))
        p -= p.mean()

        r = operators.laplacian(p, grid) - rhs
        phys_res = np.max(np.abs(r[1:-1, 1:-1]))
        print(f"Discrete residual max|lap(p) - rhs|: {phys_res:.4e}")

        return phys_res
