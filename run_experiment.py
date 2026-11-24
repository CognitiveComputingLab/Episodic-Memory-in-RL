import argparse
import importlib
import os


def run_experiment(experiment: str, index: int) -> None:
    """Import an experiment module and run the requested array element."""
    if not isinstance(index, int) or index < 0:
        raise ValueError(f"Experiment index must be a non-negative integer, got {index!r}")
    module = importlib.import_module(experiment)
    run = getattr(module, "run", None)
    if not callable(run):
        raise AttributeError(f"Experiment module {experiment!r} does not define run(index)")
    run(index)


def resolve_index(explicit: int | None) -> int:
    if explicit is not None:
        if explicit < 0:
            raise ValueError("Experiment index must be non-negative")
        return explicit

    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None:
        raise ValueError(
            "No experiment index supplied: pass --index or set SLURM_ARRAY_TASK_ID"
        )
    try:
        index = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"SLURM_ARRAY_TASK_ID must be an integer, got {raw!r}"
        ) from exc
    if index < 0:
        raise ValueError("SLURM_ARRAY_TASK_ID must be non-negative")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="e.g. experiments.tmaze")
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()
    run_experiment(args.experiment, resolve_index(args.index))


if __name__ == "__main__":
    main()