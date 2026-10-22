"""Aggregate observed replica progress and frozen charge/energy traces.

No seed pooling, phase assignment, or convergence claim is automatic. The
continuous spin angle is not a physical layer occupation observable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from mcstats import summary as trace_summary


def conservative_statistics(trace):
    result = trace_summary(trace)
    result["se"] = max(result["independent_walker_se"], result["ac"]["se"],
                       *result["block_se"].values())
    return result


def frozen(path, case):
    with np.load(path) as data:
        energies = data["energies"]
        rho = data["rho"]
        indices = data["q_indices"]
        cell = data["cell"]
    holes = case["cells"]*case["filling"]//3
    if (energies.ndim != 3 or rho.shape[:2] != energies.shape[:2] or
            not np.isfinite(energies).all() or not np.isfinite(rho).all()):
        raise ValueError(f"Invalid/nonfinite frozen trace: {path}")
    zero = np.flatnonzero(np.all(indices == 0, axis=1))
    if len(zero) != 1 or not np.allclose(rho[:, :, zero[0]], holes, atol=1e-10):
        raise ValueError(f"Particle-number sum rule failed: {path}")
    energy = conservative_statistics(energies[:, :, 0].real)
    mean = rho.mean(axis=(0, 1))
    structure = (abs(rho)**2).mean(axis=(0, 1))/holes
    coherent = abs(mean)**2/holes**2
    charge = []
    for column, index in enumerate(indices):
        z = rho[:, :, column]/holes
        real = conservative_statistics(z.real)
        imag = conservative_statistics(z.imag)
        intensity = conservative_statistics(abs(z)**2)
        charge.append(dict(q_index=index.tolist(),
            mean_rho_over_n=[float(mean[column].real/holes),
                             float(mean[column].imag/holes)],
            mean_se=[real["se"], imag["se"]],
            rank_rhat=max(real["rhat"]["rank"], imag["rhat"]["rank"],
                          intensity["rhat"]["rank"]),
            structure_factor=float(structure[column]),
            connected_structure_factor=float(structure[column]-holes*coherent[column]),
            coherent_bragg_over_n2=float(coherent[column]),
            total_bragg_over_n2=intensity["mean"],
            total_bragg_se=intensity["se"]))
    paired = path.with_suffix(".json")
    measured_summary = json.loads(paired.read_text()) if paired.exists() else None
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        records=energies.shape[0], walkers=energies.shape[1], holes=holes,
        cell=cell.tolist(), energy_mev_per_hole=energy,
        imaginary_mean_mev_per_hole=float(energies[:, :, 0].imag.mean()),
        component_means_mev_per_hole=energies.mean(axis=(0, 1)).real.tolist(),
        measured_summary=measured_summary, charge=charge)


def replica(provenance):
    metadata = json.loads(provenance.read_text())
    case = metadata["case"]
    directory = provenance.parent
    history_path = directory/"history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    interval = None
    if len(history) >= 2:
        recent = history[-min(10, len(history)):]
        eval_every = metadata["parameters"]["eval_every"]
        intervals = [(b["seconds"]-a["seconds"])/(b["step"]-a["step"])
                     for a, b in zip(recent[:-1], recent[1:])
                     if a["step"] % eval_every != 0]
        if intervals:
            interval = float(np.median(intervals))
    status_path = directory/"status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else None
    traces = sorted(directory.glob("frozen_*.npz"),
                    key=lambda p: int(p.stem.split("_")[-1]))
    measurements = [frozen(path, case) for path in traces]
    return dict(case=case, directory=str(directory), provenance=metadata,
        last_training=history[-1] if history else None,
        seconds_per_sr_step=interval, status=status, measurements=measurements,
        scientific_convergence=False,
        convergence_reason="Needs cross-seed, initial-state, capacity and size review.")


def finite_json(value):
    """Undefined diagnostics stay explicit rather than writing invalid JSON."""
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(x) for x in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    found = []
    for path in sorted(args.root.rglob("provenance.json")):
        metadata = json.loads(path.read_text())
        if "case" in metadata:
            found.append(replica(path))
    args.out.mkdir(parents=True, exist_ok=True)
    ranks = {x["case"]["rank"] for x in found}
    result = finite_json(dict(physics="R1 full continuum",
        replica_count=len(found), observed_ranks=sorted(ranks),
        missing_first_point_ranks=sorted(set(range(48))-ranks),
        replicas=found, scientific_convergence=False,
        note="Initial motif is a seed label, not a measured phase; angles are not layer occupations."))
    (args.out/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    columns = ["rank", "theta", "filling", "motif", "seed", "step",
               "seconds_per_sr_step", "frozen_mean", "frozen_se",
               "energy_rank_rhat", "mixing_certified", "converged"]
    with (args.out/"replicas.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for item in found:
            row = {key: item["case"][key] for key in columns[:5]}
            last = item["measurements"][-1] if item["measurements"] else None
            row.update(step=(item["last_training"] or {}).get("step"),
                seconds_per_sr_step=item["seconds_per_sr_step"], converged=False)
            if last:
                energy = last["energy_mev_per_hole"]
                row.update(frozen_mean=energy["mean"], frozen_se=energy["se"],
                    energy_rank_rhat=energy["rhat"]["rank"],
                    mixing_certified=(last["measured_summary"] or {}).get("mixing_certified", False))
            writer.writerow(row)
    print(json.dumps(dict(replica_count=len(found), observed_ranks=sorted(ranks),
                         summary=str(args.out/"summary.json")), indent=2))


if __name__ == "__main__":
    main()
