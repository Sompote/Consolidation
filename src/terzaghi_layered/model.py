"""Case description: layers, base condition, load history, output times.

Input files are TOML or JSON with the same keys (see examples/).  Units:
  thickness h (m), cv (m^2/yr, or m^2/day if time_unit = "day"), mv (1/kPa),
  loads dq (kPa), times in years (or days if time_unit = "day").
A layer may give mv directly, or oedometer data (e0, Cc, Cr, sigma_v0, sigma_p)
from which the secant mv over the total load increment is computed.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .core import Column, Interval, Layer

DAYS_PER_YEAR = 365.25


def secant_mv(e0: float, Cc: float, Cr: float, sigma_v0: float, sigma_p: float, dsigma: float) -> float:
    """Secant m_v (1/kPa) from the e-log(sigma') line over sigma_v0 -> sigma_v0 + dsigma."""
    if min(e0, Cc, sigma_v0, dsigma) <= 0 or Cr < 0:
        raise ValueError("oedometer data must be positive")
    sf = sigma_v0 + dsigma
    if sf <= sigma_p:
        de = Cr * np.log10(sf / sigma_v0)
    elif sigma_v0 >= sigma_p:
        de = Cc * np.log10(sf / sigma_v0)
    else:
        de = Cr * np.log10(sigma_p / sigma_v0) + Cc * np.log10(sf / sigma_p)
    return de / (1 + e0) / dsigma


@dataclass
class LoadStage:
    start: float          # yr
    end: float            # yr (== start for an instantaneous step)
    dq: float             # kPa (negative for unloading or loss of load)


@dataclass
class Case:
    title: str
    column: Column
    loads: list[LoadStage]
    times: list[float]                      # yr
    profile_times: list[float] = field(default_factory=list)
    time_unit: str = "yr"
    substeps: int = 40
    mv_source: list[str] = field(default_factory=list)

    # ------------------------------------------------------------ loading
    def steps(self) -> list[tuple[float, float]]:
        """Load history as instantaneous steps (t_k, dq_k); ramps split into substeps."""
        out = []
        for st in self.loads:
            if st.end <= st.start:
                out.append((st.start, st.dq))
            else:
                n = self.substeps
                for k in range(n):
                    out.append((st.start + (k + 0.5) * (st.end - st.start) / n, st.dq / n))
        return out

    @property
    def total_load(self) -> float:
        return float(sum(s.dq for s in self.loads))

    @property
    def final_settlement_mm(self) -> float:
        return 1000 * self.column.mass * self.total_load

    def settlement(self, t: float, tol: float = 1e-6) -> tuple[float, float, float]:
        """Settlement (mm) at time t (yr): value, lower and upper bound."""
        lo = hi = 0.0
        for tk, dq in self.steps():
            if t > tk:
                iv: Interval = self.column.degree(t - tk, tol)
                a, b = (iv.lower, iv.upper) if dq >= 0 else (iv.upper, iv.lower)
                lo += 1000 * self.column.mass * dq * a
                hi += 1000 * self.column.mass * dq * b
        return 0.5 * (lo + hi), lo, hi

    def excess_pressure(self, z, t: float) -> np.ndarray:
        """Excess pore pressure (kPa) at depths z (m) and time t (yr) under the load history."""
        u = np.zeros_like(np.atleast_1d(np.asarray(z, float)))
        for tk, dq in self.steps():
            if t >= tk:
                u += dq * self.column.excess_pressure(z, t - tk)
        return u


def _to_years(x, unit):
    return float(x) / DAYS_PER_YEAR if unit == "day" else float(x)


def load_case(path: str | Path, modes: int = 200) -> Case:
    path = Path(path)
    text = path.read_text()
    data = json.loads(text) if path.suffix.lower() == ".json" else tomllib.loads(text)
    return case_from_dict(data, modes=modes)


def case_from_dict(data: dict, modes: int = 200) -> Case:
    unit = data.get("time_unit", "yr")
    if unit not in ("yr", "day"):
        raise ValueError("time_unit must be 'yr' or 'day'")
    loads = [LoadStage(_to_years(s.get("start", 0.0), unit), _to_years(s.get("end", s.get("start", 0.0)), unit), float(s["dq"]))
             for s in data.get("load", [])]
    if not loads:
        raise ValueError("at least one [[load]] stage is required")
    dsig = sum(s.dq for s in loads if s.dq > 0)
    layers, src = [], []
    for i, L in enumerate(data.get("layer", [])):
        cv = float(L["cv"]) * (DAYS_PER_YEAR if unit == "day" else 1.0)
        if "mv" in L:
            mv = float(L["mv"]); src.append("given")
        else:
            mv = secant_mv(float(L["e0"]), float(L["Cc"]), float(L.get("Cr", 0.0)), float(L["sigma_v0"]),
                           float(L.get("sigma_p", L["sigma_v0"])), float(L.get("dsigma", dsig)))
            src.append("secant from e0, Cc, Cr, sigma_v0, sigma_p")
        layers.append(Layer(float(L["h"]), cv, mv, L.get("name", f"layer {i + 1}")))
    col = Column(layers, base=data.get("base", "sealed"), modes=modes)
    out = data.get("output", {})
    times = [_to_years(t, unit) for t in out.get("times", [0.1, 0.5, 1, 2, 5, 10, 20, 50])]
    prof = [_to_years(t, unit) for t in out.get("profile_times", [])]
    return Case(data.get("title", "layered consolidation"), col, loads, times, prof, unit,
                int(data.get("substeps", 40)), src)
