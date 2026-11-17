import numpy as np
import grid
import boundary as bnd
import initial as init
import poisson as pssn
import projection as proj
import operators as ops
import visualizer as vsl
import validation as val

if __name__ == "__main__":

    # Simulation parameters
    Re = 100.0                  # validated against Ghia et al. (1982) at Re = 100
    n_steps = 20000
    tol = 1e-6

    # Grid creation (odd number of nodes so that x = 0.5 and y = 0.5 are grid nodes)
    g = grid.Grid(129, 129, 1, 1)

    # Poisson matrix is built and LU-factorized once, then reused at every step
    poisson = pssn.PoissonSolver(g)
    poisson.test(ops, g)        # checks algebraic and discrete residuals of the solver

    # Initializing u, v, p fields and making sure they respect BCs
    u, v, p = init.apply_initial_conditions(g)
    u, v = bnd.apply_bc_velocity(u, v)

    # Creating plotting objects
    fig, axes, objects = vsl.make_figure(g)

    # Time loop (Chorin projection method)
    for n in range(n_steps):
        dt = proj.compute_dt(u, v, g, Re, 0.5)

        u_star, v_star = proj.u_intermediate(u, v, dt, Re, g)       # intermediate velocity field (pressure gradient ignored)
        u_star, v_star = bnd.apply_bc_velocity(u_star, v_star)      # applying BCs

        rhs = ops.divergence_forward(u_star, v_star, g) / dt        # rhs of the pressure Poisson equation
        p = poisson.solve(rhs)                                      # p^(n+1)

        u_new, v_new = proj.project(u_star, v_star, p, dt, g)       # projection: u^(n+1) = u* - dt * grad(p)
        u_new, v_new = bnd.apply_bc_velocity(u_new, v_new)          # applying BCs

        omega = ops.vorticity(u_new, v_new, g)                      # vorticity (only for visualization)

        diff = np.max(np.abs(u_new - u))                            # steady state check
        if diff < tol:
            print(f"Converged at step {n}, dt = {dt:.4e}")
            u, v = u_new, v_new
            break

        if n % 50 == 0:
            print(f"Step {n}: max |u^(n+1) - u^n| = {diff:.4e}")

        u, v = u_new, v_new                                         # u^(n+1) becomes u^n for the next step

        vsl.update_plots(fig, axes, objects, u, v, p, omega, g, step = n, every = 50)


    errors = val.compute_errors(*val.get_profiles(u, v, g))
    print(f"u: RMS = {errors['u_rms']:.4e}, max = {errors['u_max']:.4e}")
    print(f"v: RMS = {errors['v_rms']:.4e}, max = {errors['v_max']:.4e}")

    val.plot_comparison(u, v, g, Re = Re)

    vsl.final_plot(u, v, p, omega, g)
