import numpy as np
from numba import jit


def LB_stencil():
    """Return the D2Q9 lattice stencil and the equilibrium distribution function.

    Lattice Boltzmann Method (LBM) background
    ------------------------------------------
    Instead of tracking individual fluid particles, LBM tracks f_i(x, t): the
    probability density (or "population") of particles at lattice node x that
    are travelling in discrete direction i at time t.

    D2Q9 means: 2 spatial Dimensions, 9 discrete velocity directions Q.
    The nine directions are:
        0 — rest (stationary)
        1–4 — axis-aligned (East, North, West, South)
        5–8 — diagonal (NE, NW, SW, SE)

    Returns
    -------
    c        : (Q, 2) int array  — lattice velocity vectors, one per direction
    w        : (Q,)  float array — quadrature weights for each direction
    cs2      : float            — lattice speed of sound squared (= 1/3)
    compute_feq : callable      — evaluates the Maxwell-Boltzmann equilibrium f^eq
    """

    Q = 9
    c = np.zeros((Q, 2), dtype=np.int32)  # velocity vectors: c[i] = (cx, cy)
    w = np.zeros(Q)  # quadrature weights summing to 1

    # cs2 is the lattice speed of sound squared.  In standard D2Q9 units it is
    # always 1/3; it appears in the equilibrium distribution below.
    cs2 = 1.0 / 3.0
    qorder = 2  # order of the Gauss–Hermite quadrature (kept for reference)

    # Rest particle (direction 0) — largest weight because most particles are
    # stationary on average.
    c[0, 0] = 0
    c[0, 1] = 0
    w[0] = 4.0 / 9.0

    # Axis-aligned neighbours (directions 1–4): East, North, West, South
    c[1, 0] = 1
    c[1, 1] = 0
    w[1] = 1.0 / 9.0
    c[2, 0] = 0
    c[2, 1] = 1
    w[2] = 1.0 / 9.0
    c[3, 0] = -1
    c[3, 1] = 0
    w[3] = 1.0 / 9.0
    c[4, 0] = 0
    c[4, 1] = -1
    w[4] = 1.0 / 9.0

    # Diagonal neighbours (directions 5–8): NE, NW, SW, SE — smaller weight
    # because the effective speed (√2) is higher, so fewer particles travel there.
    c[5, 0] = 1
    c[5, 1] = 1
    w[5] = 1.0 / 36.0
    c[6, 0] = -1
    c[6, 1] = 1
    w[6] = 1.0 / 36.0
    c[7, 0] = -1
    c[7, 1] = -1
    w[7] = 1.0 / 36.0
    c[8, 0] = 1
    c[8, 1] = -1
    w[8] = 1.0 / 36.0

    # ------------------------------------------------------------------
    # Equilibrium distribution f^eq
    # ------------------------------------------------------------------
    # After a collision, populations relax towards a local Maxwell-Boltzmann
    # equilibrium.  For low Mach-number flows the second-order expansion is:
    #
    #   f^eq_i = w_i * rho * [1  +  (c_i·u)/cs²
    #                              +  (c_i·u)²/(2 cs⁴)
    #                              -  u²/(2 cs²)]
    #
    # where rho is the local density and u=(ux,uy) is the local velocity.
    # The @jit decorator (Numba) compiles this inner loop to native machine
    # code for speed, since it is called at every lattice node every timestep.
    @jit
    def compute_feq(feq, rho, ux, uy, c, w):
        # u² / cs²  — magnitude term, same for all directions
        uu = (ux**2 + uy**2) * (1.0 / cs2)

        for ip in range(Q):
            # (c_i · u) / cs²  — directional projection
            cu = (c[ip, 0] * ux[:, :] + c[ip, 1] * uy[:, :]) * (1.0 / cs2)

            # Full second-order equilibrium population
            feq[:, :, ip] = w[ip] * rho * (1.0 + cu + 0.5 * (cu * cu - uu))

        return feq

    return c, w, cs2, compute_feq
