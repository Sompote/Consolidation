"""Command-line interface: terzaghi-layered {run, quick, example}."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .core import Column, Layer
from .model import DAYS_PER_YEAR, Case, LoadStage, case_from_dict, load_case

EXAMPLES = {
    2: """title = "Two layers: soft clay over stiffer clay (paper, Section 8)"
base = "sealed"          # "sealed" or "drained" bottom boundary; top is always drained
time_unit = "yr"         # "yr" or "day" (applies to cv, load times and output times)

[[layer]]
name = "soft clay"
h  = 4.0                 # m
cv = 1.0                 # m2/yr
mv = 2.0e-3              # 1/kPa

[[layer]]
name = "stiffer clay"
h  = 6.0
cv = 4.0
mv = 0.5e-3

[[load]]                 # instantaneous step: start = end
start = 0.0
end   = 0.0
dq    = 100.0            # kPa

[output]
times = [0.5, 1, 2, 5, 10, 20, 30, 50]
profile_times = [1, 5, 20]
""",
    3: """title = "Three layers: crust, soft clay, medium clay"
base = "sealed"
time_unit = "yr"

[[layer]]
name = "weathered crust"
h  = 2.0
cv = 5.0
mv = 0.4e-3

[[layer]]
name = "soft clay"
h  = 8.0
cv = 1.2
mv = 2.5e-3

[[layer]]
name = "medium clay"
h  = 5.0
cv = 2.5
mv = 0.8e-3

[[load]]                 # ramp: fill placed over 60 days (start/end in years)
start = 0.0
end   = 0.164
dq    = 60.0

[output]
times = [0.25, 0.5, 1, 2, 5, 10, 20, 50, 100]
profile_times = [1, 10, 50]
""",
    4: """title = "Four layers with oedometer data and two load stages"
base = "drained"         # sand below the clay
time_unit = "day"        # cv in m2/day, times in days

[[layer]]
name = "crust"
h  = 1.5
cv = 0.010               # m2/day
mv = 0.3e-3              # given directly

[[layer]]
name = "very soft clay"
h  = 4.0
cv = 0.003
e0 = 2.8                 # mv computed as secant from oedometer data
Cc = 1.2
Cr = 0.12
sigma_v0 = 18.0          # kPa, at mid-layer
sigma_p  = 25.0          # kPa

[[layer]]
name = "soft clay"
h  = 6.0
cv = 0.004
e0 = 2.2
Cc = 0.9
Cr = 0.09
sigma_v0 = 40.0
sigma_p  = 52.0

[[layer]]
name = "medium clay"
h  = 4.0
cv = 0.008
mv = 0.5e-3

[[load]]
start = 0
end   = 30
dq    = 35.0

[[load]]
start = 180
end   = 210
dq    = 25.0

[output]
times = [30, 90, 180, 210, 365, 730, 1825, 3650]
profile_times = [210, 730]
""",
}


def _fmt_time(t_yr: float, unit: str) -> str:
    return f"{t_yr * DAYS_PER_YEAR:.1f} d" if unit == "day" else f"{t_yr:.3g} yr"


def report(case: Case, tol: float = 1e-6, csv_path: str | None = None, plot_path: str | None = None,
           out=None) -> dict:
    out = sys.stdout if out is None else out
    col, unit = case.column, case.time_unit
    w = out.write
    w(f"\n{case.title}\n" + "=" * len(case.title) + "\n")
    w(f"top drained, base {col.base}; {col.n} layers, total thickness {col.zb[-1]:.2f} m\n\n")
    w(f"{'layer':<18}{'depth (m)':>14}{'cv (m2/yr)':>12}{'mv (1/kPa)':>12}{'l (yr^.5)':>11}{'e':>11}{'mv*h share':>12}\n")
    for i, L in enumerate(col.layers):
        w(f"{L.name[:17]:<18}{col.zb[i]:>6.2f}-{col.zb[i+1]:<7.2f}{L.cv:>12.4g}{L.mv:>12.4g}{col.ell[i]:>11.4g}"
          f"{col.e[i]:>11.3e}{L.mv * L.h / col.mass:>11.1%}\n")
    if any(s != "given" for s in case.mv_source):
        w("  (mv: " + "; ".join(f"{L.name}: {s}" for L, s in zip(col.layers, case.mv_source) if s != "given") + ")\n")
    w(f"\ncolumn diffusion time (sum l)^2 = {col.ell.sum() ** 2:.4g} yr\n")
    w(f"total load {case.total_load:.4g} kPa, final primary settlement {case.final_settlement_mm:.1f} mm\n")
    if len(case.loads) == 1 and case.loads[0].end <= case.loads[0].start:
        t50, t90 = col.time_to(0.5), col.time_to(0.9)
        w(f"single step: t50 = {_fmt_time(t50, unit)}, t90 = {_fmt_time(t90, unit)}\n")
    else:
        t50 = t90 = None
        w("staged load: t50/t90 refer to a single step and are not reported; see the table\n")
    w(f"\n{'time':>12}{'settlement (mm)':>17}{'lower':>11}{'upper':>11}{'U = s/s_f':>11}\n")
    rows = []
    for t in case.times:
        v, lo, hi = case.settlement(t, tol)
        U = v / case.final_settlement_mm if case.final_settlement_mm else float("nan")
        rows.append(dict(time=t * (DAYS_PER_YEAR if unit == "day" else 1), time_unit=unit, settlement_mm=v,
                         lower_mm=lo, upper_mm=hi, U=U))
        flag = "" if case.tolerance_met(t, tol) else "  tolerance not met"
        w(f"{_fmt_time(t, unit):>12}{v:>17.2f}{lo:>11.2f}{hi:>11.2f}{U:>11.4f}{flag}\n")
    w(f"(bounds: certified truncation interval of the exact series, tolerance {tol:g} per load step)\n")
    profiles = []
    if case.profile_times:
        z = np.linspace(0, col.zb[-1], 11)
        w(f"\nexcess pore pressure (kPa)\n{'depth (m)':>10}" + "".join(f"{_fmt_time(t, unit):>12}" for t in case.profile_times) + "\n")
        U = np.array([case.excess_pressure(z, t) for t in case.profile_times])
        for j, zz in enumerate(z):
            w(f"{zz:>10.2f}" + "".join(f"{U[i, j]:>12.2f}" for i in range(len(case.profile_times))) + "\n")
        profiles = U.tolist()
    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
        w(f"\nwrote {csv_path}\n")
    if plot_path:
        _plot(case, plot_path, tol)
        w(f"wrote {plot_path}\n")
    return dict(rows=rows, t50=t50, t90=t90, profiles=profiles)


def _plot(case: Case, path: str, tol: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col, unit = case.column, case.time_unit
    k = DAYS_PER_YEAR if unit == "day" else 1.0
    tmax = max(case.times) * 1.2
    T = np.geomspace(tmax / 2000, tmax, 160)
    s = np.array([case.settlement(t, 1e-4)[0] for t in T])
    ncol = 2 if case.profile_times else 1
    fig, ax = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.2), constrained_layout=True, squeeze=False)
    a = ax[0, 0]
    a.semilogx(T * k, s, "k-", lw=1.6, label="exact layered series")
    tt = np.array(case.times)
    a.semilogx(tt * k, [case.settlement(t, tol)[0] for t in tt], "ro", ms=4, label="reported times")
    a.axhline(case.final_settlement_mm, color="0.5", ls="--", lw=0.8, label="final primary settlement")
    a.set(xlabel=f"time ({unit})", ylabel="settlement (mm)", title=case.title[:60])
    a.invert_yaxis(); a.grid(True, which="both", alpha=0.3); a.legend(frameon=False, fontsize=8)
    if case.profile_times:
        a = ax[0, 1]; z = np.linspace(0, col.zb[-1], 200)
        for t in case.profile_times:
            a.plot(case.excess_pressure(z, t), z, lw=1.4, label=_fmt_time(t, unit))
        for zb in col.zb[1:-1]:
            a.axhline(zb, color="0.6", lw=0.6, ls=":")
        a.set(xlabel="excess pore pressure (kPa)", ylabel="depth (m)", title="isochrones")
        a.invert_yaxis(); a.grid(True, alpha=0.3); a.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _parse_layer(s: str) -> dict:
    parts = s.split(",")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError("--layer expects h,cv,mv[,name]")
    d = dict(h=float(parts[0]), cv=float(parts[1]), mv=float(parts[2]))
    if len(parts) == 4:
        d["name"] = parts[3]
    return d


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="terzaghi-layered",
                                description="Exact 1D consolidation of layered soil (any number of layers) "
                                            "with certified error bounds.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a case file (.toml or .json)")
    r.add_argument("case")
    r.add_argument("--tol", type=float, default=1e-6, help="truncation tolerance on U per load step (default 1e-6)")
    r.add_argument("--modes", type=int, default=200, help="eigenmodes to compute (default 200)")
    r.add_argument("--csv", help="write the settlement table to CSV")
    r.add_argument("--plot", help="write settlement and isochrone plot (PNG/PDF)")

    q = sub.add_parser("quick", help="one load step, layers on the command line")
    q.add_argument("--layer", action="append", type=_parse_layer, required=True,
                   help="h,cv,mv[,name] in m, m2/yr, 1/kPa (top to bottom; repeat)")
    q.add_argument("--load", type=float, required=True, help="load step at t=0 (kPa)")
    q.add_argument("--base", choices=["sealed", "drained"], default="sealed")
    q.add_argument("--times", type=float, nargs="+", default=[0.5, 1, 2, 5, 10, 20, 50], help="years")
    q.add_argument("--tol", type=float, default=1e-6)
    q.add_argument("--csv"); q.add_argument("--plot")

    e = sub.add_parser("example", help="print an example case file")
    e.add_argument("--layers", type=int, choices=[2, 3, 4], default=3)

    a = p.parse_args(argv)
    try:
        if a.cmd == "example":
            sys.stdout.write(EXAMPLES[a.layers]); return 0
        if a.cmd == "run":
            case = load_case(a.case, modes=a.modes)
            report(case, a.tol, a.csv, a.plot); return 0
        if a.cmd == "quick":
            case = case_from_dict(dict(title="quick case", base=a.base, layer=a.layer,
                                       load=[dict(start=0, end=0, dq=a.load)], output=dict(times=a.times)))
            report(case, a.tol, a.csv, a.plot); return 0
    except (ValueError, KeyError, FileNotFoundError) as exc:
        sys.stderr.write(f"error: {exc}\n"); return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
