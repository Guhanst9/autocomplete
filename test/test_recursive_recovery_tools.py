import csv

from compare_recovery_pilot import compare, summarize
from compare_recursive_benchmarks import select_candidate


def benchmark(device, hourly_price, times):
    return {
        "device": device,
        "hourly_price": hourly_price,
        "results": [
            {
                "mode": mode,
                "estimated_epoch_seconds": seconds,
                "estimated_epoch_cost": seconds / 3600 * hourly_price,
            }
            for mode, seconds in times.items()
        ],
    }


def test_benchmark_selection_rejects_slow_48_base_mode():
    a10g = benchmark(
        "A10G",
        1.006,
        {"independent": 3000, "recursive-24": 3600, "recursive-48": 4200},
    )
    l40s = benchmark(
        "L40S",
        1.861,
        {"independent": 1800, "recursive-24": 2100, "recursive-48": 2200},
    )
    result = select_candidate([a10g, l40s])

    assert not any(
        row["device"] == "A10G" and row["mode"] == "recursive-48"
        for row in result["candidates"]
    )
    assert result["selected"]["mode"] in {"recursive-24", "recursive-48"}


def write_eval(path, generated, truth):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["generated_suffix", "true_suffix"])
        writer.writeheader()
        for _ in range(614):
            writer.writerow({"generated_suffix": generated, "true_suffix": truth})


def test_pilot_summary_and_acceptance(tmp_path):
    control_path = tmp_path / "control.csv"
    experiment_path = tmp_path / "experiment.csv"
    truth = "A" * 512
    write_eval(control_path, "C" * 12 + "A" * 500, truth)
    write_eval(experiment_path, "C" * 5 + "A" * 507, truth)

    control = summarize(control_path)
    experiment = summarize(experiment_path)
    result = compare(control, experiment)

    assert experiment["accuracy_percent"] > control["accuracy_percent"]
    assert result["checks"]["final_100_improved_by_1pp"] is False
    assert result["checks"]["no_runs_over_20"] is False
    assert result["pass"] is False
