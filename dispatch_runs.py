import argparse, subprocess
from run_experiment import run_experiment

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("experiment")
    p.add_argument("n", type=int)
    p.add_argument("--max-concurrent", type=int, default=4)
    p.add_argument("--time", default="00:30:00")
    p.add_argument("--no-slurm", action="store_true")
    args = p.parse_args()

    if not args.no_slurm:
        name = args.experiment.split(".")[-1]
        cmd = ["sbatch",
        f"--array=0-{(args.n)-1}%{args.max_concurrent}",
        f"--job-name={name}",
        f"--time={args.time}",
        "run_experiment.sh", args.experiment]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        for i in range(args.n):
            run_experiment(args.experiment, i)
        