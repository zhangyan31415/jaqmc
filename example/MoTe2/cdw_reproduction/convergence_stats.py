"""Independent ArviZ diagnostics for frozen, fixed-parameter walker traces."""
from __future__ import annotations

import arviz_stats as az
import numpy as np

from mcstats import summary as legacy_summary


def trace_statistics(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or min(x.shape) < 4 or not np.isfinite(x).all():
        raise ValueError("Expected finite (records, independent walkers) traces.")
    legacy = legacy_summary(x)
    chains = x.T
    independent_se = legacy["independent_walker_se"]
    mcse = float(az.mcse(chains, method="mean"))
    return dict(mean=float(x.mean()), variance=float(x.var()),
        rank_rhat=float(az.rhat(chains, method="rank")),
        legacy_rank_rhat=legacy["rhat"]["rank"],
        bulk_ess=float(az.ess(chains, method="bulk")),
        tail_ess=float(az.ess(chains, method="tail", prob=.05)),
        arviz_mcse=mcse, independent_walker_se=independent_se,
        se=max(independent_se, mcse, legacy["ac"]["se"],
               *legacy["block_se"].values()),
        tau=legacy["ac"]["tau"], records=x.shape[0], walkers=x.shape[1])


def difference_check(first, second, sigma=3., tolerance=.01):
    a, b = trace_statistics(first), trace_statistics(second)
    difference = b["mean"]-a["mean"]
    se = float(np.hypot(a["se"], b["se"]))
    return dict(difference=difference, se=se,
        z=abs(difference)/max(se, 1e-30),
        passed=bool(abs(difference) <= sigma*se+tolerance))


def observable_checks(x, rhat_limit, tolerance=.01, sigma=3.):
    x = np.asarray(x)
    stats = trace_statistics(x)
    half_walkers, half_records = x.shape[1]//2, x.shape[0]//2
    families = difference_check(x[:, :half_walkers], x[:, half_walkers:],
                               sigma=sigma, tolerance=tolerance)
    drift = difference_check(x[:half_records], x[half_records:],
                            sigma=sigma, tolerance=tolerance)
    finite = all(np.isfinite(stats[key]) for key in
                 ["rank_rhat", "bulk_ess", "tail_ess", "se"])
    passed = (finite and stats["rank_rhat"] < rhat_limit and
              min(stats["bulk_ess"], stats["tail_ess"]) >= 400 and
              families["passed"] and drift["passed"])
    return dict(statistics=stats, family_comparison=families,
                half_comparison=drift, passed=bool(passed))


def diagnose(energies, rho, q_indices, holes):
    energies, rho = np.asarray(energies), np.asarray(rho)
    if not np.isfinite(energies).all() or not np.isfinite(rho).all():
        raise FloatingPointError("Nonfinite frozen observables.")
    zero = np.flatnonzero(np.all(np.asarray(q_indices) == 0, axis=1))
    if len(zero) != 1 or not np.allclose(rho[:, :, zero[0]], holes, atol=1e-10):
        raise ValueError("Charge particle-number sum rule failed.")
    energy = observable_checks(energies[:, :, 0].real, 1.01)
    imag = trace_statistics(energies[:, :, 0].imag)
    hermitian = abs(imag["mean"]) <= 5*imag["se"]+.01
    charge = []
    for index, q in enumerate(q_indices):
        if np.all(np.asarray(q) == 0):
            continue
        z = rho[:, :, index]/holes
        checks = {label: observable_checks(x, 1.02, tolerance=.001, sigma=4.)
                  for label, x in [("real", z.real), ("imag", z.imag),
                                   ("abs", abs(z))]}
        charge.append(dict(q_index=np.asarray(q).tolist(), checks=checks,
                           passed=all(c["passed"] for c in checks.values())))
    return dict(energy=energy, imaginary_energy=imag,
        hermitian_mean_pass=bool(hermitian), charge=charge,
        mixing_pass=bool(energy["passed"] and hermitian and
                        all(q["passed"] for q in charge)),
        diagnostic_backend="arviz-stats 1.3.3",
        energy_rank_rhat_limit=1.01, charge_rank_rhat_limit=1.02,
        minimum_bulk_and_tail_ess=400,
        scientific_convergence=False)


def plateau(previous, final, tolerance=.1, max_se=.1):
    a = previous["energy"]["statistics"]
    b = final["energy"]["statistics"]
    delta = b["mean"]-a["mean"]
    se = float(np.hypot(a["se"], b["se"]))
    valid = previous["mixing_pass"] and final["mixing_pass"]
    passed = valid and max(a["se"], b["se"]) <= max_se and abs(delta) <= 2*se+tolerance
    return dict(delta_mev_per_hole=delta, difference_se=se,
        tolerance_mev_per_hole=tolerance, maximum_se_mev_per_hole=max_se,
        frozen_measurements_valid=bool(valid),
        fixed_capacity_plateau_pass=bool(passed), scientific_convergence=False,
        note="Capacity, system size, and competing-branch checks remain separate.")
