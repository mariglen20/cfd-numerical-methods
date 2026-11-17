import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded

# Grid convergence study: the numerical solution is compared with an exact solution
# on a sequence of refined grids, and the observed order of accuracy is
#     p = log(e_coarse / e_fine) / log(h_coarse / h_fine)
#
# Test 1: 2D Poisson, -lap(p) = f on [0, 1]^2, p = 0 on the boundary
#         exact solution p = sin(pi x) sin(pi y)  ->  f = 2 pi^2 sin(pi x) sin(pi y)
#         5-point stencil, expected order 2 in space
#
# Test 2: 1D heat equation, u_t = alpha * u_xx on [0, 1], u(0) = u(1) = 0
#         exact solution u = exp(-alpha pi^2 t) sin(pi x)
#         a) explicit FTCS with fixed r = alpha dt / dx^2  -> dt ~ dx^2, expected order 2 in dx
#         b) implicit Euler with fixed dt / dx             -> dt ~ dx, the O(dt) time error dominates: expected order 1

def error_norms(e):
    """Discrete error norms: RMS (L2 normalized by the number of nodes) and max (L-infinity)."""
    return np.sqrt(np.mean(e**2)), np.max(np.abs(e))

def observed_order(h, err):
    h = np.asarray(h)
    err = np.asarray(err)
    return np.log(err[:-1] / err[1:]) / np.log(h[:-1] / h[1:])


# ---------------- Test 1: 2D Poisson ----------------

def poisson_exact(X, Y):
    return np.sin(np.pi * X) * np.sin(np.pi * Y)

def poisson_forcing(X, Y):
    return 2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

def solve_poisson(n):
    """
    Solves -lap(p) = f on an n x n grid (boundary nodes included).
    Unknowns are only the (n-2)^2 interior nodes, boundary values are 0.
    The 2D matrix is built from the 1D second difference matrix: A = I (x) D2 + D2 (x) I.
    """
    h = 1 / (n - 1)
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x)

    m = n - 2
    D2 = diags([-np.ones(m - 1), 2 * np.ones(m), -np.ones(m - 1)], [-1, 0, 1]) / h**2
    I = identity(m)
    A = (kron(I, D2) + kron(D2, I)).tocsr()

    b = poisson_forcing(X[1:-1, 1:-1], Y[1:-1, 1:-1]).ravel()

    p = np.zeros((n, n))
    p[1:-1, 1:-1] = spsolve(A, b).reshape((m, m))
    return h, p - poisson_exact(X, Y)


# ---------------- Test 2: 1D heat equation ----------------

def heat_exact(x, t, alpha):
    return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)

def heat_explicit(nx, alpha, T, r):
    x = np.linspace(0, 1, nx)
    dx = x[1] - x[0]
    nt = int(np.ceil(T / (r * dx**2 / alpha)))
    dt = T / nt                                 # dt adjusted to land exactly on T
    r = alpha * dt / dx**2

    u = heat_exact(x, 0, alpha)
    for _ in range(nt):
        u[1:-1] = u[1:-1] + r * (u[:-2] - 2 * u[1:-1] + u[2:])
    return dx, u - heat_exact(x, T, alpha)

def heat_implicit(nx, alpha, T, dt_over_dx):
    x = np.linspace(0, 1, nx)
    dx = x[1] - x[0]
    nt = int(np.ceil(T / (dt_over_dx * dx)))
    dt = T / nt
    r = alpha * dt / dx**2

    n = nx - 2
    ab = np.zeros((3, n))
    ab[0, 1:] = -r
    ab[1, :] = 1 + 2 * r
    ab[2, :-1] = -r

    u = heat_exact(x, 0, alpha)
    for _ in range(nt):
        u[1:-1] = solve_banded((1, 1), ab, u[1:-1])     # homogeneous Dirichlet BCs: no boundary terms
    return dx, u - heat_exact(x, T, alpha)


def run_study(solver, sizes, *args):
    h, rms, emax = [], [], []
    for n in sizes:
        hn, e = solver(n, *args)
        e_rms, e_max = error_norms(e)
        h.append(hn)
        rms.append(e_rms)
        emax.append(e_max)
    return np.array(h), np.array(rms), np.array(emax)

def print_table(name, h, rms, emax):
    print(f"\n{name}")
    print(f"{'h':>10} {'RMS error':>12} {'order':>6} {'max error':>12} {'order':>6}")
    p_rms = observed_order(h, rms)
    p_max = observed_order(h, emax)
    for i in range(len(h)):
        o1 = f"{p_rms[i-1]:6.2f}" if i > 0 else f"{'-':>6}"
        o2 = f"{p_max[i-1]:6.2f}" if i > 0 else f"{'-':>6}"
        print(f"{h[i]:10.5f} {rms[i]:12.4e} {o1} {emax[i]:12.4e} {o2}")


def plot_study(results, filename):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))

    for ax, (title, h, rms, emax, order) in zip(axes, results):
        ax.loglog(h, rms, "o-", label="RMS error")
        ax.loglog(h, emax, "s-", label="max error")
        ref = emax[0] * (h / h[0])**order                  # reference slope through the first point
        ax.loglog(h, ref, "k--", label=f"slope {order}")
        ax.set_xlabel("h")
        ax.set_ylabel("error")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


if __name__ == "__main__":
    alpha = 1.0
    T = 0.1

    h1, rms1, max1 = run_study(solve_poisson, [9, 17, 33, 65, 129, 257])
    h2, rms2, max2 = run_study(heat_explicit, [11, 21, 41, 81, 161], alpha, T, 0.4)
    h3, rms3, max3 = run_study(heat_implicit, [11, 21, 41, 81, 161], alpha, T, 0.5)

    print_table("2D Poisson, 5-point stencil", h1, rms1, max1)
    print_table("1D heat, explicit FTCS (r = 0.4)", h2, rms2, max2)
    print_table("1D heat, implicit Euler (dt = 0.5 dx)", h3, rms3, max3)

    plot_study([
        ("2D Poisson, 5-point stencil", h1, rms1, max1, 2),
        ("1D heat, explicit FTCS, r = 0.4", h2, rms2, max2, 2),
        ("1D heat, implicit Euler, dt = 0.5 dx", h3, rms3, max3, 1),
    ], "convergence.png")
