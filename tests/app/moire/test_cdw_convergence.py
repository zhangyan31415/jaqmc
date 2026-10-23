"""Check rejection of inconsistent banks and unvalidated plateau claims."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] /
                      "example/MoTe2/cdw_reproduction"))
from convergence_stats import difference_check, diagnose, plateau, trace_statistics


def test_independent_reference_and_displaced_bank():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(1024, 8))
    reference = trace_statistics(x)
    assert reference["rank_rhat"] < 1.01
    assert reference["bulk_ess"] > 400
    assert reference["tail_ess"] > 400
    assert abs(reference["rank_rhat"]-reference["legacy_rank_rhat"]) < 1e-12
    assert difference_check(x[:, :4], x[:, 4:])["passed"]
    shifted = x.copy()
    shifted[:, 4:] += 3.
    assert trace_statistics(shifted)["rank_rhat"] > 1.1
    assert not difference_check(shifted[:, :4], shifted[:, 4:])["passed"]


def test_plateau_requires_valid_mixing_precision_and_stability():
    def measured(mean=0., se=.01, mixed=True):
        return dict(mixing_pass=mixed,
                    energy=dict(statistics=dict(mean=mean, se=se)))
    assert plateau(measured(), measured(.02))["fixed_capacity_plateau_pass"]
    assert not plateau(measured(mixed=False), measured())["fixed_capacity_plateau_pass"]
    assert not plateau(measured(), measured(-.5))["fixed_capacity_plateau_pass"]
    assert not plateau(measured(se=.3), measured(se=.3))["fixed_capacity_plateau_pass"]


def test_nonfinite_and_wrong_particle_count_rejected():
    with pytest.raises(ValueError, match="finite"):
        trace_statistics(np.full((16, 4), np.nan))
    energy = np.ones((16, 4, 5), dtype=complex)
    rho = np.full((16, 4, 1), 8., dtype=complex)
    with pytest.raises(ValueError, match="particle-number"):
        diagnose(energy, rho, np.array([[0, 0]]), holes=9)


def test_subset_launcher_queues_ranks_and_reads_correct_original_hosts(tmp_path, monkeypatch):
    import launch_revalidation as launch
    calls = []

    class FinishedProcess:
        returncode = 0

        def poll(self):
            return 0

    def spawn(command, **kwargs):
        calls.append((command, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        return FinishedProcess()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    monkeypatch.setattr(launch.os, "sched_getaffinity", lambda _: set(range(6)), raising=False)
    monkeypatch.setattr(launch.subprocess, "Popen", spawn)
    monkeypatch.setattr(launch.time, "sleep", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["launch_revalidation.py", "--host", "0",
        "--checkpoint-root", str(tmp_path/"original"),
        "--out", str(tmp_path/"out"), "--remote", str(tmp_path/"remote"),
        "--ranks", "0,12,36,24"])
    launch.main()
    assert [int(command[command.index("--rank")+1]) for command, _ in calls] == [0, 12, 36, 24]
    assert [gpu for _, gpu in calls] == ["0", "1", "2", "0"]
    assert [Path(command[command.index("--checkpoint-root")+1]).name
            for command, _ in calls] == ["host0", "host1", "host4", "host3"]
    import json
    status = json.loads((tmp_path/"remote/host_status.json").read_text())
    assert status["execution_finished"] and status["failure"] is None
    assert status["requested_ranks"] == [0, 12, 36, 24]
