import argparse, importlib, os
from train import train

def run_experiment(experiment: str, index: int):
    train(importlib.import_module(experiment).getRun(index))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("experiment")                    # e.g. experiments.xmaze_corridors
    p.add_argument("--index", type=int, default=None)
    args = p.parse_args()
    idx = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    run_experiment(args.experiment, idx)
