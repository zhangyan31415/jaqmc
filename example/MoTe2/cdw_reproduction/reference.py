"""Independent first-shell electron/hole convention and CDW geometry checks.

Source: arXiv:2603.05714v2, SM S1--S6, and arXiv:2304.11864v2.
The axes and periodic layer gauge below are explicitly chosen to match JAQMC.
This is the full continuum model (R1), not the paper's adiabatic LL model.
"""
from __future__ import annotations

import numpy as np

from jaqmc.app.moire.config import MoireConfig
from jaqmc.app.moire.free_reference import DEFAULT_LATTICE, reciprocal_vectors
from jaqmc.app.moire.model import MatrixContinuumModel

SITES = {"MM": (0., 0.), "MX": (1/3, 2/3), "XM": (2/3, 1/3)}
PAPER = dict(v_mev=20.8, w_electron_mev=-23.8, psi_deg=107.7,
             mass=0.6, epsilon=10., lattice_nm=0.352)


def configuration(theta: float, filling: int, cells: int = 27) -> MoireConfig:
    if cells not in (3, 12, 27) or filling not in (1, 2):
        raise ValueError("Use 3,12,27 cells and filling numerator 1 or 2 over 3.")
    matrix = {3: [[2, 1], [1, 2]], 12: [[4, 2], [2, 4]],
              27: [[6, 3], [3, 6]]}[cells]
    count = cells * filling // 3
    return MoireConfig(nelec=count, electron_spins=(count, 0),
        supercell_matrix=matrix, effective_mass=PAPER["mass"],
        dielectric_constant=PAPER["epsilon"], v1_mev=PAPER["v_mev"],
        phi1_deg=PAPER["psi_deg"],
        # h_h(p) = -h_e(-p)^T: tunneling changes sign as well.
        omega1_mev=-PAPER["w_electron_mev"],
        moire_lattice_constant_nm=PAPER["lattice_nm"] /
            (2 * np.sin(np.deg2rad(theta)/2)))


def electron_fields(r: np.ndarray) -> np.ndarray:
    """S6 in rotated axes and periodic gauge; no JAQMC potential calls."""
    b1, b2 = reciprocal_vectors(DEFAULT_LATTICE)
    angles = r @ np.array([b1, b2-b1, -b2]).T
    psi = np.deg2rad(PAPER["psi_deg"])
    d = 2 * PAPER["v_mev"] * np.cos(angles[..., None] +
        np.array([psi, -psi])).sum(axis=-2)
    # The conjugation follows from reversing momenta in the hole basis.
    t = PAPER["w_electron_mev"] * (1 + np.exp(-1j*(r@b2)) +
                                             np.exp(-1j*(r@(b2-b1))))
    h = np.zeros((*r.shape[:-1], 2, 2), dtype=complex)
    h[..., 0, 0], h[..., 1, 1] = d[..., 0], d[..., 1]
    h[..., 0, 1], h[..., 1, 0] = t, t.conj()
    return h


def hole_model(cfg: MoireConfig) -> MatrixContinuumModel:
    """Build independently from -h_e(-p)^T, including valley shifts."""
    b1, b2 = reciprocal_vectors(DEFAULT_LATTICE)
    k = np.array([(b1+b2)/3, (2*b1-b2)/3])
    t = cfg.kinetic_prefactor_mev / (2*cfg.effective_mass)
    constant = np.diag(t*np.sum(k*k, axis=1)).astype(complex)
    constant[0, 1] = constant[1, 0] = -PAPER["w_electron_mev"]
    powers = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [0, 2]])
    matrices = np.array([constant, np.diag(-2*t*k[:, 0]),
        np.diag(-2*t*k[:, 1]), t*np.eye(2), t*np.eye(2)])
    psi = np.deg2rad(PAPER["psi_deg"])
    fourier = np.tile(np.diag(-PAPER["v_mev"] *
                              np.exp(1j*np.array([psi, -psi]))), (3, 1, 1))
    # Store g=(b1,b2-b1,-b2), one vector per +/- pair.
    fourier[1, 0, 1] = -PAPER["w_electron_mev"]
    fourier[2, 1, 0] = -PAPER["w_electron_mev"]
    return MatrixContinuumModel(powers, matrices, np.array([b1, b2-b1, -b2]),
        fourier, basis_labels=("bottom-hole", "top-hole"),
        name="Wang-2024 first shell; full continuum R1; Eref=0")


def motif_centers(cfg: MoireConfig, motif: str) -> np.ndarray:
    """One occupied site at 1/3; two occupied sites at 2/3.

    At 2/3 the motif names the depleted site, so MM means MX+XM holes.
    The offsets are an initialization, never a constraint on the Hamiltonian.
    MX/XM are conventional labels for the two layer-exchanged sites in this
    choice of axes; switching valley or layers interchanges them.
    """
    if motif not in SITES:
        raise ValueError(motif)
    matrix = np.asarray(cfg.supercell_matrix)
    inverse = np.linalg.inv(matrix)
    origins = []
    bound = int(np.max(np.abs(matrix))*3+3)
    for u in range(-bound, bound+1):
        for v in range(-bound, bound+1):
            if (u+v) % 3:
                continue
            fraction = (np.array([u, v]) @ inverse) % 1.
            fraction[np.isclose(fraction, 1.)] = 0.
            if not any(np.allclose(fraction, x) for x in origins):
                origins.append(fraction)
    assert len(origins) == cfg.scale//3
    occupied = [motif] if cfg.nelec == cfg.scale//3 else [
        label for label in SITES if label != motif]
    centers = [(f + np.array(SITES[label]) @ inverse) % 1.
               for f in origins for label in occupied]
    return np.asarray(centers) @ cfg.supercell_lattice


def spectral_seed(cfg: MoireConfig, motif: str, sigma: float = .3,
                  cutoff: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Periodic Gaussian Slater orbitals in a trainable Fourier basis."""
    centers = motif_centers(cfg, motif)
    b = reciprocal_vectors(cfg.supercell_lattice)
    indices = np.array([(i, j) for i in range(-cutoff, cutoff+1)
                       for j in range(-cutoff, cutoff+1)])
    g = indices @ b
    envelope = np.exp(-.5*sigma**2*np.sum(g*g, axis=1))
    coeff = envelope[:, None] * np.exp(-1j*g@centers.T)
    # Choose the local low-energy hole spinor separately at every orbital.
    _, vectors = np.linalg.eigh(-electron_fields(centers).swapaxes(-1, -2))
    spinor = vectors[..., 0]
    coeff = np.einsum("gj,jl->lgj", coeff, spinor)
    norm = np.sqrt(np.sum(abs(coeff)**2, axis=(0, 1)))
    coeff /= norm[None, None, :]
    return g, coeff, centers


def exact_slater(cfg: MoireConfig, cutoff: int = 3):
    """Occupied free eigenstates, with every folded momentum represented."""
    from jaqmc.app.moire.free_reference import folded_k_points, q_shell
    q = q_shell(DEFAULT_LATTICE, cutoff)
    kpoints = folded_k_points(DEFAULT_LATTICE, np.array(cfg.supercell_matrix),
                             np.array(cfg.twist))
    model = hole_model(cfg)
    states = []
    spectra = []
    for index, k in enumerate(kpoints):
        values, vectors = np.linalg.eigh(model.plane_wave_hamiltonian(k, q))
        spectra.append(values)
        states.extend((float(values[band]), index, band, vectors[:, band])
                      for band in range(min(cfg.nelec, len(values))))
    occupied = sorted(states, key=lambda x: x[0])[:cfg.nelec]
    g = np.concatenate([k+q for k in kpoints])
    coeff = np.zeros((2, len(g), cfg.nelec), dtype=complex)
    for column, (_, index, _, vector) in enumerate(occupied):
        coeff[:, index*len(q):(index+1)*len(q), column] = vector.reshape(-1, 2).T
    return g, coeff, sum(x[0] for x in occupied), spectra
