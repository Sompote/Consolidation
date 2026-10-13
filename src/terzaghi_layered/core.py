"""Exact one-dimensional consolidation of a layered column.

Linear layered model (each layer: thickness h, coefficient of consolidation c_v,
compressibility m_v), drained top, sealed or drained base, uniform initial excess
pressure (a load step).  In diffusion coordinates x_i = z/sqrt(c_v,i) every layer
is a unit-diffusivity segment of length l_i = h_i/sqrt(c_v,i) and the interfaces
carry only the effusivity e_i = m_v,i sqrt(c_v,i).

Two certified evaluations of the degree of settlement consolidation U_s(t):
  * spectral: U_s = 1 - sum w_n exp(-lambda_n t) with w_n >= 0, sum w_n = 1, so
    keeping k modes gives a two-sided interval (late and intermediate times);
  * reflection paths (sealed base only): erfc series with a graph-resolvent bound
    on every omitted path (early times).
The hybrid rule takes the spectral interval when few modes suffice, otherwise the
path series.  Eigenvalues come from a monotone unwrapped (Pruefer) phase, so no
root can be missed.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import erfc


@dataclass
class Layer:
    """One soil layer. Units: h in m, cv in m^2/yr, mv in 1/kPa."""
    h: float
    cv: float
    mv: float
    name: str = ""

    def __post_init__(self):
        for k in ("h", "cv", "mv"):
            v = getattr(self, k)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{self.name or 'layer'}: {k} must be positive, got {v}")


@dataclass
class Interval:
    lower: float
    upper: float
    method: str
    terms: int

    @property
    def value(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def half_width(self) -> float:
        return 0.5 * (self.upper - self.lower)


class Column:
    """Layered column with drained top and 'sealed' or 'drained' base."""

    def __init__(self, layers: list[Layer], base: str = "sealed", modes: int = 200):
        if base not in ("sealed", "drained"):
            raise ValueError("base must be 'sealed' or 'drained'")
        if not layers:
            raise ValueError("at least one layer is required")
        self.layers = list(layers)
        self.base = base
        self.h = np.array([l.h for l in layers], float)
        self.cv = np.array([l.cv for l in layers], float)
        self.mv = np.array([l.mv for l in layers], float)
        self.d = self.cv * self.mv                      # D = k / gamma_w
        self.n = len(layers)
        self.ell = self.h / np.sqrt(self.cv)            # diffusion lengths (yr^1/2)
        self.e = self.mv * np.sqrt(self.cv)             # effusivities
        self.mass = float(np.dot(self.mv, self.h))      # sum m_v h  (m/kPa)
        self.zb = np.concatenate([[0.0], np.cumsum(self.h)])
        self._build_scattering()
        self._eigen(modes)

    # ------------------------------------------------------------------ setup
    def _build_scattering(self):
        n, e = self.n, self.e
        S = np.zeros((2 * n, 2 * n))
        S[0, 1] = -1.0
        S[-1, -2] = 1.0 if self.base == "sealed" else -1.0
        for i in range(n - 1):
            r = (e[i] - e[i + 1]) / (e[i] + e[i + 1])
            tau = 2 * np.sqrt(e[i] * e[i + 1]) / (e[i] + e[i + 1])
            S[2 * i + 1, 2 * i] = r
            S[2 * i + 1, 2 * i + 3] = tau
            S[2 * i + 2, 2 * i] = tau
            S[2 * i + 2, 2 * i + 3] = -r
        self.S = S
        self.absS = np.abs(S)
        self.lengths = np.repeat(self.ell, 2)
        self.theta0 = max(np.log(np.sum(self.absS, axis=0)) / self.lengths)
        self.edges = [[(int(k), S[k, j]) for k in np.flatnonzero(S[:, j])] for j in range(2 * n)]

    def phase(self, w: float) -> float:
        """Monotone unwrapped phase at the base for wavenumber w (yr^-1/2)."""
        theta = 0.0
        for i in range(self.n):
            theta += w * self.ell[i]
            if i < self.n - 1:
                turns = np.floor(theta / np.pi)
                rem = theta - turns * np.pi
                theta = turns * np.pi + np.arctan2(self.e[i + 1] / self.e[i] * np.sin(rem), np.cos(rem))
        return theta

    def _eigen(self, modes: int):
        omega = []
        L = self.ell.sum()
        for j in range(modes):
            target = (j + (0.5 if self.base == "sealed" else 1.0)) * np.pi
            lo = omega[-1] if omega else 0.0
            hi = max((target + self.n * np.pi) / L, lo + 1 / L)
            while self.phase(hi) < target:
                hi *= 2
            omega.append(brentq(lambda w: self.phase(w) - target, lo, hi, xtol=2e-13, rtol=1e-14))
        self.omega = np.asarray(omega)
        self.lam = self.omega ** 2
        self.coef = []          # per mode, per layer (A, B) of phi = A cos(b z') + B sin(b z')
        weights, norms, proj = [], [], []
        for w in self.omega:
            value, flux = 0.0, 1.0
            I1 = I2 = 0.0
            ab = []
            for h, d, m in zip(self.h, self.d, self.mv):
                b = w * np.sqrt(m / d)
                ang = b * h
                co, si = np.cos(ang), np.sin(ang)
                A, B = value, flux / (d * b)
                ab.append((A, B, b))
                I1 += m * (A * si + B * 2 * np.sin(ang / 2) ** 2) / b
                I2 += m * (A * A * (h / 2 + np.sin(2 * ang) / (4 * b))
                           + B * B * (h / 2 - np.sin(2 * ang) / (4 * b)) + A * B * si * si / b)
                value, flux = A * co + B * si, d * b * (-A * si + B * co)
            self.coef.append(ab)
            weights.append(I1 ** 2 / (self.mass * I2))
            norms.append(I2)
            proj.append(I1)
        self.weights = np.array(weights)
        self.norms = np.array(norms)
        self.proj = np.array(proj)
        if np.any(self.weights < 0) or self.weights.sum() > 1 + 1e-9:
            raise RuntimeError("modal weights failed the positivity check")

    # ------------------------------------------------------------ spectral
    def reference(self, t: float) -> float:
        """U_s(t) from all computed modes (accurate unless t is very small)."""
        if t <= 0:
            return 0.0
        return float(1 - np.dot(self.weights, np.exp(-self.lam * t)))

    def spectral_interval(self, t: float, k: int) -> tuple[float, float]:
        remain = max(0.0, 1 - self.weights[:k].sum())
        upper = 1 - np.dot(self.weights[:k], np.exp(-self.lam[:k] * t))
        lower = upper - remain * np.exp(-self.lam[k] * t)
        return max(0.0, lower), min(1.0, upper)

    def required_modes(self, t: float, tol: float):
        for k in range(len(self.lam) - 1):
            lo, hi = self.spectral_interval(t, k)
            if (hi - lo) / 2 <= tol:
                return k
        return None

    # ------------------------------------------------------- reflection paths
    def _advance(self, wave):
        nxt = defaultdict(float)
        for (j, counts), c in wave.items():
            cnt = list(counts)
            cnt[j // 2] += 1
            for k, a in self.edges[j]:
                nxt[(k, tuple(cnt))] += a * c
        return {k: v for k, v in nxt.items() if v != 0.0}

    def _arrays(self, wave):
        st = np.array([k[0] for k in wave], int)
        L = np.array([np.dot(k[1], self.ell) for k in wave])
        c = np.array(list(wave.values()))
        return st, L, c

    def _contribution(self, wave, t):
        st, L, c = self._arrays(wave)
        def F(y):
            return y * erfc(y / (2 * np.sqrt(t))) - 2 * np.sqrt(t / np.pi) * np.exp(-y * y / (4 * t))
        li = self.ell[st // 2]
        return float(np.sum(np.sqrt(self.e[0] * self.e[st // 2]) * c * (F(L + li) - F(L))) / self.mass)

    def _graph_bound(self, frontier, t):
        if not frontier:
            return 0.0
        st, L, c = self._arrays(frontier)
        Lmin = L.min()
        I = np.eye(2 * self.n)
        def obj(theta):
            B = self.absS * np.exp(-theta * self.lengths)[None, :]
            src = np.bincount(st, weights=np.abs(c) * np.exp(-theta * (L - Lmin)), minlength=2 * self.n)
            desc = np.linalg.solve(I - B, src)
            term = np.repeat(np.sqrt(self.e[0] * self.e) * (-np.expm1(-theta * self.ell)) / theta / self.mass, 2)
            mag = np.dot(term, desc)
            if mag <= 0 or not np.isfinite(mag):
                return np.inf
            return theta * theta * t - theta * Lmin + np.log(mag)
        lo = max(self.theta0 * (1 + 1e-7), 1e-9)
        hi = max(2 * lo, Lmin / t + 4 / np.sqrt(t))
        opt = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-8})
        val = min(obj(x) for x in (lo, opt.x, hi))
        return float(np.exp(max(-690.0, min(val, 700.0))))

    def paths(self, t: float, tol: float, maxterms: int = 20000):
        if self.base != "sealed":
            return None
        wave = {(0, (0,) * self.n): 1.0}
        value, terms = 0.0, 0
        while terms + len(wave) <= maxterms:
            terms += len(wave)
            value += self._contribution(wave, t)
            wave = self._advance(wave)
            bound = self._graph_bound(wave, t)
            if bound <= tol:
                return value, bound, terms
            if not wave:
                break
        return None

    # ---------------------------------------------------------------- public
    def degree(self, t: float, tol: float = 1e-6) -> Interval:
        """Certified U_s(t) (single load step at t=0), t in years."""
        if t <= 0:
            return Interval(0.0, 0.0, "t<=0", 0)
        k = self.required_modes(t, tol)
        if k is not None and k <= 8:
            lo, hi = self.spectral_interval(t, k)
            return Interval(lo, hi, "spectral", k)
        p = self.paths(t, tol)
        if p is not None:
            v, b, terms = p
            return Interval(max(0.0, v - b), min(1.0, v + b), "paths", terms)
        if k is not None:
            lo, hi = self.spectral_interval(t, k)
            return Interval(lo, hi, "spectral", k)
        lo, hi = self.spectral_interval(t, len(self.lam) - 1)
        return Interval(lo, hi, "spectral (tolerance not met)", len(self.lam) - 1)

    def time_to(self, U: float) -> float:
        """Time (years) to reach degree U for a single load step."""
        if not 0 < U < 1:
            raise ValueError("U must be in (0, 1)")
        hi = 1.0
        while self.degree(hi).value < U:
            hi *= 2
        return float(brentq(lambda t: self.degree(t, 1e-9).value - U, 1e-12, hi, xtol=1e-10, rtol=1e-10))

    def excess_pressure(self, z, t: float) -> np.ndarray:
        """Excess pore pressure u/Q at depths z (m) for a unit load step, from the modal series."""
        z = np.atleast_1d(np.asarray(z, float))
        out = np.zeros_like(z)
        if t <= 0:
            out[:] = 1.0
            out[z <= 0] = 0.0
            return out
        idx = np.clip(np.searchsorted(self.zb, z, side="right") - 1, 0, self.n - 1)
        zl = z - self.zb[idx]
        amp = self.proj / self.norms * np.exp(-self.lam * t)
        for n_, (a, ab) in enumerate(zip(amp, self.coef)):
            if abs(a) < 1e-16:
                continue
            A = np.array([x[0] for x in ab])[idx]
            B = np.array([x[1] for x in ab])[idx]
            b = np.array([x[2] for x in ab])[idx]
            out += a * (A * np.cos(b * zl) + B * np.sin(b * zl))
        return out
