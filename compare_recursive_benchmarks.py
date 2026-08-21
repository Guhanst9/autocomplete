import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def select_candidate(benchmarks):
    candidates = []
    for benchmark in benchmarks:
        modes = {row["mode"]: row for row in benchmark["results"]}
        allowed = [modes["recursive-24"]]
        if modes["recursive-48"]["estimated_epoch_seconds"] <= (
            1.15 * modes["recursive-24"]["estimated_epoch_seconds"]
        ):
            allowed.append(modes["recursive-48"])
        for row in allowed:
            candidates.append(
                {
                    "device": benchmark["device"],
                    "mode": row["mode"],
                    "epoch_seconds": row["estimated_epoch_seconds"],
                    "epoch_cost": row["estimated_epoch_cost"],
                }
            )

    lowest_cost = min(candidates, key=lambda row: row["epoch_cost"])
    fastest = min(candidates, key=lambda row: row["epoch_seconds"])
    runtime_reduction = 1.0 - fastest["epoch_seconds"] / lowest_cost["epoch_seconds"]
    cost_increase = fastest["epoch_cost"] / lowest_cost["epoch_cost"] - 1.0
    selected = fastest if runtime_reduction >= 0.35 and cost_increase <= 0.20 else lowest_cost
    return {
        "selected": selected,
        "lowest_cost": lowest_cost,
        "fastest": fastest,
        "fastest_runtime_reduction": runtime_reduction,
        "fastest_cost_increase": cost_increase,
        "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Select the recursive recovery GPU and block size."
    )
    parser.add_argument("benchmarks", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.benchmarks) < 2:
        parser.error("provide at least two benchmark JSON files")
    result = select_candidate([load(path) for path in args.benchmarks])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["selected"], indent=2))


if __name__ == "__main__":
    main()
