import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def benchmark_binary(build_dir):
    candidates = [build_dir / "bin" / "celiums-bitnet"]
    if platform.system() == "Windows":
        candidates.insert(0, build_dir / "bin" / "Release" / "celiums-bitnet.exe")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    installed = shutil.which("celiums-bitnet")
    if installed:
        return Path(installed)
    raise FileNotFoundError("celiums-bitnet not found")


def build_command(args):
    command = [
        str(benchmark_binary(args.build_dir)), "bench",
        "-m", str(args.model),
        "--model-family", getattr(args, "model_family", "bitnet"),
        "-p", str(args.n_prompt),
        "-n", str(args.n_token),
        "-b", str(args.batch),
        "-ub", str(args.ubatch),
        "-t", str(args.threads),
        "-r", str(args.repetitions),
    ]
    if getattr(args, "n_seq", None):
        command.extend(["--n-seq", str(args.n_seq)])
    if getattr(args, "ram_budget_bytes", None):
        command.extend(["--ram-budget-bytes", str(args.ram_budget_bytes)])
    if getattr(args, "compute_layout", None) is not None:
        command.extend(["--compute-layout", "1" if args.compute_layout else "0"])
    if args.output != "jsonl":
        raise ValueError("The native runtime benchmark currently supports only jsonl output")
    if args.cpu_mask:
        raise ValueError("CPU masks are not yet available in the native runtime benchmark")
    return command


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Celiums BitNet prefill and decode")
    parser.add_argument("-m", "--model", type=Path, required=True)
    parser.add_argument("--model-family", choices=["bitnet", "bonsai"], default="bitnet")
    parser.add_argument("-n", "--n-token", type=int, default=128)
    parser.add_argument("-p", "--n-prompt", type=int, default=128)
    parser.add_argument("-t", "--threads", type=int, default=2)
    parser.add_argument("-b", "--batch", type=int, default=128)
    parser.add_argument("-ub", "--ubatch", type=int, default=128)
    parser.add_argument("-r", "--repetitions", type=int, default=5)
    parser.add_argument("-o", "--output", choices=["csv", "json", "jsonl", "md", "sql"], default="jsonl")
    parser.add_argument("--cpu-mask")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--n-seq", type=int, default=1)
    parser.add_argument("--ram-budget-bytes", type=int, default=0)
    parser.add_argument("--compute-layout", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    args.build_dir = args.build_dir.resolve()
    return args


if __name__ == "__main__":
    try:
        subprocess.run(build_command(parse_args()), cwd=ROOT, check=True)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)
