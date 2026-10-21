#!/usr/bin/env python3
"""Verify the archived selected M-valley examples using NumPy only.

No source-machine paths, fitting, native TAPW calculation or JAQMC import is
required. Archived overlaps are checked as data, not recomputed from native
wavefunctions (the source export verification records that separate check).
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import socket
import sys
from urllib.parse import unquote, urlsplit

import numpy as np


TAG = "seven_parameter_constant_mass_20261021"


def load_model(directory, name):
    spec = importlib.util.spec_from_file_location(name, directory / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(example_root):
    repo = example_root.parent
    manifest = json.loads((example_root / "continuum_models_manifest.json").read_text())
    archived = [r for r in manifest["copied_files"] if f"/models/M/{TAG}/" in r["destination"]]
    assert len(archived) == 66, f"Expected 66 copied files; found {len(archived)}"
    for row in archived:
        path = repo / row["destination"]
        assert path.stat().st_size == row["bytes"], path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path

    reports = []
    for name, active in [("SnS2", [0, 1, 2, 3, 4, 7, 8]), ("SnSe2", [0, 1, 2, 3, 6, 7, 8])]:
        directory = example_root / name / "AA/models/M" / TAG
        model = load_model(directory, "archive_" + name)
        config = model.CONFIG
        selection = json.loads((directory / "reports/SELECTION.json").read_text())
        assert config["active_full9_indices"] == active
        assert config["physical_parameter_count"] == 7
        assert config["global_coefficient_count"] == 17
        assert len(model.CATALOG) == 12
        controls = np.asarray(config["global_controls"])
        assert controls.shape == (7, 3)
        np.testing.assert_allclose(controls[:2], np.repeat(controls[:2, :1], 3, axis=1), atol=1e-13)
        np.testing.assert_allclose(
            model.parameters(np.linspace(*config["angle_endpoint_deg"], 37))[:2],
            np.repeat(controls[:2, :1], 37, axis=1), atol=1e-12,
        )
        errors = dict(spectrum_eV=0.0, hermiticity_eV=0.0, potential_periodicity_eV=0.0,
                      tensor_eigenvalues_eV_A2=0.0, parameter_eV_or_eV_A2=0.0)
        rows = []
        for record, fitted in zip(model.CATALOG, config["records"]):
            theta = record["theta_deg"]
            packet = np.load(directory / "reports/predictions" / (record["case"] + ".npz"))
            fractional = packet["fractional"]
            assert fractional.shape == (61, 2)
            np.testing.assert_allclose(fractional[-1], fractional[0], atol=1e-13)
            p = model.parameters(theta)
            errors["parameter_eV_or_eV_A2"] = max(errors["parameter_eV_or_eV_A2"], float(np.max(abs(p - fitted["parameters"]))))
            full = model.full_parameters(theta)
            assert np.all(full[[i for i in range(9) if i not in active]] == 0)
            tensors = model.kinetic_tensors(theta)
            assert np.linalg.norm(tensors[0] - tensors[1]) > 1e-8, "Actual layer axes were lost"
            errors["tensor_eigenvalues_eV_A2"] = max(errors["tensor_eigenvalues_eV_A2"],
                float(np.max(abs(np.linalg.eigvalsh(tensors) - np.sort(p[:2])))))
            geometry = model.geometry(theta)
            enlarged = model.enlarged_cell(theta)
            original = 2 * np.pi * np.linalg.inv(geometry["B"]).T
            np.testing.assert_allclose(abs(np.linalg.det(enlarged)), 2 * abs(np.linalg.det(original)), rtol=1e-12)
            positions = np.random.default_rng(42).normal(size=(8, 2)) * 30
            potential = model.realspace(theta, positions)
            for shift in enlarged:
                errors["potential_periodicity_eV"] = max(errors["potential_periodicity_eV"],
                    float(np.max(abs(potential - model.realspace(theta, positions + shift)))))
            zero = np.linalg.eigvalsh(model.hamiltonian(theta, [0.0, 0.0]))[0]
            for index, k in enumerate(fractional):
                matrix = model.hamiltonian(theta, k)
                errors["hermiticity_eV"] = max(errors["hermiticity_eV"], float(np.max(abs(matrix - matrix.conj().T))))
                energy = np.linalg.eigvalsh(matrix)[:5] - zero
                errors["spectrum_eV"] = max(errors["spectrum_eV"],
                    float(np.max(abs(np.repeat(energy, 2) - packet["model_energy"][index]))))
            probabilities = packet["group_probability"][:60]
            assert probabilities.shape == (60, 3)
            assert np.isfinite(probabilities).all() and probabilities.min() >= -1e-12
            assert probabilities.max() <= 1 + 1e-10
            energy_error = (packet["model_pair_centers"][:60] - packet["native_pair_centers"][:60]) * 1000
            rows.append(dict(theta=theta, rms=np.sqrt(np.mean(energy_error**2, axis=0)), overlap=probabilities.min(axis=0)))
        for bound, reference in selection["scope"]["scopes"].items():
            subset = [row for row in rows if row["theta"] <= float(bound)]
            assert len(subset) == reference["angles"]
            rms = np.sqrt(np.mean([row["rms"]**2 for row in subset], axis=0))
            overlap = np.min([row["overlap"] for row in subset], axis=0)
            np.testing.assert_allclose(rms, reference["pair_RMS_meV"], atol=1e-10)
            np.testing.assert_allclose(overlap, reference["P_by_pair_min"], atol=1e-12)
            np.testing.assert_allclose(np.sqrt(np.mean(rms[:2]**2)), reference["first_two_pair_RMS_meV"], atol=1e-10)

        # Independent real-space Fourier reconstruction at the 3.89-degree case.
        theta = model.record(3.89)["theta_deg"]
        geometry = model.geometry(theta)
        points = np.array([(i / 24, j / 24) for i in range(24) for j in range(24)]) @ model.enlarged_cell(theta)
        potential = model.realspace(theta, points)
        q = [geometry["q1"], geometry["q2"]]
        offsets = np.cumsum([0, len(q[0]), len(q[1])])
        tensors = model.kinetic_tensors(theta)
        fourier_error = 0.0
        for k in (np.array([0.0, 0.0]), np.array([0.19, -0.11])):
            matrix = np.zeros((offsets[-1], offsets[-1]), complex)
            for layer in range(2):
                for other in range(2):
                    delta = q[layer][:, None] - q[other][None, :]
                    field = potential[:, layer] if layer == other else potential[:, 2]
                    coefficients = np.einsum("rij,r->ij", np.exp(1j * np.einsum("ija,ra->rij", delta, points)), field) / len(points)
                    matrix[offsets[layer]:offsets[layer + 1], offsets[other]:offsets[other + 1]] = coefficients
                p = k @ geometry["B"] - q[layer]
                matrix[offsets[layer]:offsets[layer + 1], offsets[layer]:offsets[layer + 1]] += np.diag(np.einsum("ia,ab,ib->i", p, tensors[layer], p))
            fourier_error = max(fourier_error, float(np.max(abs(matrix - model.hamiltonian(theta, k)))))
        assert max(errors.values()) < 1e-9, errors
        assert fourier_error < 1e-10, fourier_error
        reports.append(dict(material=name,angles=12,path_rows_per_angle=61,independent_k_per_angle=60,
                            errors=errors,fourier_error_eV=fourier_error,
                            retained_original_scientific_status=config["status"]))
        print("PASS", name, json.dumps(reports[-1]), flush=True)

    # Check links in the new docs and central index, independently of source paths.
    markdown = [example_root / "README.md", example_root / "M_VALLEY_SEVEN_PARAMETER.md"]
    for name in ("SnS2", "SnSe2"):
        markdown.extend([example_root/name/"README.md",example_root/name/"AA/README.md",
                         example_root/name/"AA/models/M"/TAG/"README.md"])
    links = 0
    for path in markdown:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            assert (path.parent / unquote(parsed.path)).exists(), (str(path), target)
            links += 1
    return dict(status="passed",scope="Portable archive checks only; not new fitting, native wavefunction validation or ground-state training",
                copied_files_verified=len(archived),readme_links_checked=links,materials=reports,
                python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,
                hostname=socket.gethostname(),working_directory=str(repo),command=[sys.executable,*sys.argv],
                process_thread_layout="one process; BLAS/OpenMP threads=1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-root",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--out",type=Path)
    arguments = parser.parse_args()
    result = verify(arguments.example_root.resolve())
    if arguments.out:
        arguments.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result,indent=2,ensure_ascii=False))
