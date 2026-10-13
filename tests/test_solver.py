import numpy as np
import pytest

from terzaghi_layered import Column, Layer, case_from_dict, secant_mv
from terzaghi_layered.cli import EXAMPLES, main
from fd_reference import fd_degree


def terzaghi(T, n=400):
    M = (2 * np.arange(n) + 1) * np.pi / 2
    return 1 - np.sum(2 / M ** 2 * np.exp(-M ** 2 * T))


def test_homogeneous_matches_terzaghi():
    c = Column([Layer(10, 2, 1e-3)])
    for T in (0.01, 0.1, 0.5, 1.0):
        assert c.degree(T * 100 / 2).value == pytest.approx(terzaghi(T), abs=1e-6)
    assert c.time_to(0.5) * 2 / 100 == pytest.approx(0.196731, abs=1e-6)
    assert c.time_to(0.9) * 2 / 100 == pytest.approx(0.848085, abs=1e-6)


def test_split_homogeneous_layer_is_unchanged():
    one = Column([Layer(10, 2, 1e-3)])
    three = Column([Layer(3, 2, 1e-3), Layer(4, 2, 1e-3), Layer(3, 2, 1e-3)])
    for t in (0.5, 5, 30):
        assert three.degree(t).value == pytest.approx(one.degree(t).value, abs=2e-6)


def test_drained_base_equals_half_thickness():
    full = Column([Layer(10, 1, 1e-3)], base="drained")
    half = Column([Layer(5, 1, 1e-3)], base="sealed")
    for t in (0.2, 2, 10):
        assert full.degree(t).value == pytest.approx(half.degree(t).value, abs=2e-6)


def test_paper_two_layer_example():
    c = Column([Layer(4, 1, 2e-3), Layer(6, 4, 0.5e-3)])
    assert c.degree(10).value == pytest.approx(0.632181, abs=2e-6)
    assert c.time_to(0.9) == pytest.approx(27.6504, abs=1e-3)
    assert c.time_to(0.5) == pytest.approx(6.0117, abs=1e-3)
    assert c.weights[0] == pytest.approx(0.7631, abs=1e-4)


@pytest.mark.parametrize("base", ["sealed", "drained"])
@pytest.mark.parametrize("profile", [
    ([2.0, 8.0, 5.0], [5.0, 1.2, 2.5], [0.4e-3, 2.5e-3, 0.8e-3]),
    ([1.5, 4.0, 6.0, 4.0], [3.6, 1.1, 1.5, 2.9], [0.3e-3, 2.7e-3, 1.4e-3, 0.5e-3]),
    ([1.0, 2.0, 3.0, 2.0], [10.0, 0.1, 2.0, 0.05], [0.2e-3, 3e-3, 1e-3, 2e-3]),
])
def test_three_and_four_layers_against_finite_differences(profile, base):
    h, cv, mv = profile
    c = Column([Layer(*x) for x in zip(h, cv, mv)], base=base)
    L2 = c.ell.sum() ** 2
    times = np.array([1e-3, 1e-2, 0.1, 0.5]) * L2
    fd = fd_degree(h, cv, mv, times, base=base)
    ex = np.array([c.degree(t).value for t in times])
    assert np.max(np.abs(ex - fd)) < 5e-5


def test_interval_contains_reference_and_meets_tolerance():
    c = Column([Layer(1.5, 3.6, .3e-3), Layer(4, 1.1, 2.7e-3), Layer(6, 1.5, 1.4e-3), Layer(4, 2.9, .5e-3)], modes=600)
    for t in np.geomspace(1e-4, 3, 12) * c.ell.sum() ** 2:
        iv = c.degree(t, 1e-6)
        assert iv.half_width <= 1e-6 + 1e-12
        assert iv.lower - 1e-12 <= c.reference(t) <= iv.upper + 1e-12 or iv.method == "paths"


def test_pressure_profile_boundaries_and_settlement_consistency():
    c = Column([Layer(2, 5, .4e-3), Layer(8, 1.2, 2.5e-3), Layer(5, 2.5, .8e-3)])
    z = np.linspace(0, 15, 3001)
    for t in (1.0, 10.0):
        u = c.excess_pressure(z, t)
        assert abs(u[0]) < 1e-6 and 0 <= u.min() + 1e-6 and u.max() <= 1 + 1e-3
        m = np.interp(z, [0, 2, 2, 10, 10, 15], [.4e-3, .4e-3, 2.5e-3, 2.5e-3, .8e-3, .8e-3])
        Us = 1 - np.trapz(m * u, z) / c.mass
        assert Us == pytest.approx(c.degree(t).value, abs=2e-4)


def test_staged_load_is_superposition():
    base = dict(base="sealed", layer=[dict(h=4, cv=1, mv=2e-3), dict(h=6, cv=4, mv=.5e-3)], output=dict(times=[5]))
    staged = case_from_dict(dict(base, load=[dict(start=0, end=0, dq=60), dict(start=2, end=2, dq=40)]))
    c = staged.column
    expect = 1000 * c.mass * (60 * c.degree(5).value + 40 * c.degree(3).value)
    assert staged.settlement(5)[0] == pytest.approx(expect, abs=1e-6)
    assert expect == pytest.approx(458.17, abs=0.01)          # paper, Appendix C, Step 7


def test_ramp_tends_to_step():
    d = dict(layer=[dict(h=10, cv=1, mv=1e-3)], output=dict(times=[1]))
    ramp = case_from_dict(dict(d, load=[dict(start=0, end=1e-4, dq=100)]))
    step = case_from_dict(dict(d, load=[dict(start=0, end=0, dq=100)]))
    assert ramp.settlement(1)[0] == pytest.approx(step.settlement(1)[0], rel=1e-4)


def test_secant_mv_branches():
    assert secant_mv(2, 1, .1, 20, 20, 30) == pytest.approx(1 * np.log10(50 / 20) / 3 / 30)
    assert secant_mv(2, 1, .1, 20, 60, 30) == pytest.approx(.1 * np.log10(50 / 20) / 3 / 30)
    oc = .1 * np.log10(30 / 20) + 1 * np.log10(50 / 30)
    assert secant_mv(2, 1, .1, 20, 30, 30) == pytest.approx(oc / 3 / 30)


def test_invalid_input_rejected():
    with pytest.raises(ValueError):
        Layer(1, -1, 1e-3)
    with pytest.raises(ValueError):
        Column([Layer(1, 1, 1e-3)], base="open")


@pytest.mark.parametrize("n", [2, 3, 4])
def test_cli_examples_run(n, tmp_path, capsys):
    f = tmp_path / f"case{n}.toml"
    f.write_text(EXAMPLES[n])
    assert main(["run", str(f), "--csv", str(tmp_path / "o.csv")]) == 0
    out = capsys.readouterr().out
    assert "final primary settlement" in out and (tmp_path / "o.csv").exists()


def test_cli_quick_reproduces_paper(capsys):
    assert main(["quick", "--layer", "4,1,2e-3", "--layer", "6,4,0.5e-3", "--load", "100", "--times", "10"]) == 0
    assert "695.40" in capsys.readouterr().out
