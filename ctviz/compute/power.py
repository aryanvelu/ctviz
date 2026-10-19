"""Power of a two-sided, two-sample z-test for a standardized effect size (Cohen's d).

Normal approximation, ignoring the negligible opposite tail:
``power(d, n) = Phi(d * sqrt(n / 2) - z_{1-alpha/2})`` with ``n`` participants per arm.
At d = 0 this equals alpha/2 (one tail), which keeps the curve monotone in d and n.
"""

from __future__ import annotations

import math

ALPHA = 0.05
TARGET_POWER = 0.8
ASSUMPTIONS = [
    "Two-sided alpha = 0.05, two-sample z-test (normal approximation; opposite tail ignored).",
    "Continuous endpoint with equal variance; effect size is standardized (Cohen's d).",
    "Equal allocation: n per arm (unequal arms use the harmonic mean of the arm sizes).",
    "No dropout, multiplicity, interim-analysis or non-inferiority adjustments.",
    "This is not the trial's registered power calculation (the registry does not report one).",
]


def phi(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def inv_phi(p: float) -> float:
    """Inverse standard normal CDF by bisection (|error| < 1e-10)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if phi(mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return (lo + hi) / 2


def z_crit(alpha: float = ALPHA) -> float:
    return inv_phi(1.0 - alpha / 2.0)


def power(d: float, n_per_arm: float, alpha: float = ALPHA, *, one_sample: bool = False) -> float:
    """Approximate power for effect size ``d`` with ``n_per_arm`` participants per arm."""
    if n_per_arm <= 0:
        return 0.0
    se_factor = math.sqrt(n_per_arm) if one_sample else math.sqrt(n_per_arm / 2.0)
    return phi(abs(d) * se_factor - z_crit(alpha))


def mde(n_per_arm: float, target: float = TARGET_POWER, alpha: float = ALPHA, *, one_sample: bool = False) -> float:
    """Minimum detectable standardized effect at ``target`` power."""
    if n_per_arm <= 0:
        return math.inf
    k = math.sqrt(n_per_arm) if one_sample else math.sqrt(n_per_arm / 2.0)
    return (z_crit(alpha) + inv_phi(target)) / k


def harmonic_mean(values: list[float]) -> float:
    vals = [v for v in values if v > 0]
    return len(vals) / sum(1.0 / v for v in vals) if vals else 0.0


def effect_grid(max_d: float = 1.0, step: float = 0.025) -> list[float]:
    n = int(round(max_d / step))
    return [round(i * step, 4) for i in range(n + 1)]


def power_curve(
    n_per_arm: float, grid: list[float] | None = None, alpha: float = ALPHA, *, one_sample: bool = False
) -> list[tuple[float, float]]:
    """``[(d, power)]`` over the effect-size grid."""
    return [(d, round(power(d, n_per_arm, alpha, one_sample=one_sample), 4)) for d in grid or effect_grid()]
