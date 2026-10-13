"""Independent check: cell-centred Crank-Nicolson finite differences (Rannacher start-up,
harmonic interface conductances) for m u_t = (D u_z)_z, drained top, sealed or drained base,
u(z,0) = 1.  Shares no code with the package."""
import numpy as np
from scipy.linalg import solve_banded


def fd_degree(h, cv, mv, times, base="sealed", cells=400, steps_per_decade=2000):
    h, cv, mv = (np.asarray(a, float) for a in (h, cv, mv))
    d = cv * mv
    dz = np.repeat(h / cells, cells); dd = np.repeat(d, cells); mm = np.repeat(mv, cells)
    cond = 1 / (dz[:-1] / (2 * dd[:-1]) + dz[1:] / (2 * dd[1:]))
    diag = np.zeros(len(dz)); diag[:-1] += cond; diag[1:] += cond
    diag[0] += 2 * dd[0] / dz[0]
    if base == "drained":
        diag[-1] += 2 * dd[-1] / dz[-1]
    mass = mm * dz; n = len(dz)
    tmax = max(times); t0 = 1e-9 * tmax
    grid = np.unique(np.concatenate([np.geomspace(t0, tmax, int(np.log10(tmax / t0) * steps_per_decade) + 1), times]))
    def band(a):
        ab = np.zeros((3, n)); ab[1] = mass + a * diag; ab[0, 1:] = -a * cond; ab[2, :-1] = -a * cond; return ab
    def Au(v):
        r = diag * v; r[:-1] -= cond * v[1:]; r[1:] -= cond * v[:-1]; return r
    u = np.ones(n); t = 0.0; out = {}
    for k, tn in enumerate(grid):
        dt = tn - t
        if k < 2:
            for _ in range(2):
                u = solve_banded((1, 1), band(dt / 2), mass * u)
        else:
            u = solve_banded((1, 1), band(dt / 2), mass * u - dt / 2 * Au(u))
        t = tn
        out[float(tn)] = 1 - np.dot(mass, u) / mass.sum()
    return np.array([out[float(x)] for x in times])
