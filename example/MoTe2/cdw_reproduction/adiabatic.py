"""R0 scalar/texture diagnostics from arXiv:2603.05714v2, SM S18--S27.

This constructs the adiabatic fields, flux, and first-shell LLL potential.
It is not a self-consistent multilevel Hartree--Fock calculation.
"""
from __future__ import annotations

import numpy as np
from reference import PAPER, configuration


def diagnostics(theta: float, grid: int = 128, first_shell: bool = False) -> dict:
    cfg = configuration(theta, 1, 3)
    lattice = np.asarray(cfg.lattice_vectors)
    b1, b2 = cfg.reciprocal_lattice
    axis = np.arange(grid)/grid
    u, v = np.meshgrid(axis, axis, indexing="ij")
    r = np.stack([u, v], axis=-1)@lattice
    shell = np.array([b1, b2-b1, -b2])
    q1 = (2*b2-b1)/3
    transfers = np.array([q1, q1-b2, q1-b2+b1])
    projection = r@shell.T
    psi = np.deg2rad(PAPER["psi_deg"])
    bottom = 2*PAPER["v_mev"]*np.cos(projection+psi).sum(axis=-1)
    top = 2*PAPER["v_mev"]*np.cos(projection-psi).sum(axis=-1)
    bottom_grad = -2*PAPER["v_mev"]*np.einsum(
        "...g,gd->...d", np.sin(projection+psi), shell)
    top_grad = -2*PAPER["v_mev"]*np.einsum(
        "...g,gd->...d", np.sin(projection-psi), shell)
    exponential = np.exp(1j*r@transfers.T)
    tunneling = PAPER["w_electron_mev"]*exponential.sum(axis=-1)
    tunneling_grad = 1j*PAPER["w_electron_mev"]*np.einsum(
        "...g,gd->...d", exponential, transfers)
    delta = np.stack([tunneling.real, -tunneling.imag, (bottom-top)/2], axis=-1)
    derivative = np.stack([tunneling_grad.real, -tunneling_grad.imag,
                          (bottom_grad-top_grad)/2], axis=-2)
    magnitude = np.linalg.norm(delta, axis=-1)
    n = delta/magnitude[..., None]
    dn = (derivative-n[..., :, None]*np.einsum("...l,...ld->...d", n, derivative)[
        ..., None, :])/magnitude[..., None, None]
    curvature = .5*np.einsum("...l,...l->...", n,
                            np.cross(dn[..., 0], dn[..., 1]))
    metric = np.sum(dn*dn, axis=(-1, -2))/4
    kinetic = cfg.kinetic_prefactor_mev/(2*cfg.effective_mass)
    u_ad = (bottom+top)/2 + magnitude-kinetic*metric
    spectrum = np.fft.fft2(curvature-curvature.mean())/(grid*grid)
    f = np.fft.fftfreq(grid)*grid
    indices = np.stack(np.meshgrid(f, f, indexing="ij"), axis=-1)
    g = indices@cfg.reciprocal_lattice
    squared = np.sum(g*g, axis=-1)
    if first_shell:
        length = np.linalg.norm(b1)
        spectrum *= np.isclose(np.sqrt(squared), length)
        curvature = (np.fft.ifft2(spectrum*grid*grid).real + curvature.mean())
    squared[0, 0] = 1
    ax = 1j*g[..., 1]*spectrum/squared
    ay = -1j*g[..., 0]*spectrum/squared
    ax = np.fft.ifft2(ax*grid*grid).real
    ay = np.fft.ifft2(ay*grid*grid).real
    # The ladder convention changes with the orientation of the mean flux.
    electron_v = (u_ad - kinetic*np.sign(curvature.mean()) *
                  (curvature-curvature.mean()) - kinetic*(ax*ax+ay*ay))
    hole_coeff = -np.fft.fft2(electron_v)/(grid*grid)
    harmonics = np.array([hole_coeff[1, 0], hole_coeff[-1, 1], hole_coeff[0, -1]])
    area = abs(np.linalg.det(lattice))
    return dict(theta=theta, grid=grid, flux_quanta=float(curvature.mean()*area/(2*np.pi)),
        hole_first_shell_mev=float(harmonics.real.mean()),
        first_shell_imag_max=float(abs(harmonics.imag).max()),
        first_shell_spread=float(np.ptp(harmonics.real)),
        min_pseudospin_splitting_mev=float(2*magnitude.min()),
        metric_mean=float(metric.mean()), first_shell_fields=first_shell,
        model="R0 fields only; no HF solution")
