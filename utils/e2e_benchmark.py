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
        "-p", str(args.n_prompt),
        "-n", str(args.n_token),
        "-b", str(args.batch),
        "-ub", str(args.ubatch),
        "-t", str(args.threads),
        "-r", str(args.repetitions),
        "-o", args.output,
        "-ngl", "0",
    ]
    if args.cpu_mask:
        command.extend(["-C", args.cpu_mask, "--cpu-strict", "1"])
    return command


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Celiums BitNet prefill and decode")
    parser.add_argument("-m", "--model", type=Path, required=True)
    parser.add_argument("-n", "--n-token", type=int, default=128)
    parser.add_argument("-p", "--n-prompt", type=int, default=128)
    parser.add_argument("-t", "--threads", type=int, default=2)
    parser.add_argument("-b", "--batch", type=int, default=128)
    parser.add_argument("-ub", "--ubatch", type=int, default=128)
    parser.add_argument("-r", "--repetitions", type=int, default=5)
    parser.add_argument("-o", "--output", choices=["csv", "json", "jsonl", "md", "sql"], default="jsonl")
    parser.add_argument("--cpu-mask")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    args.build_dir = args.build_dir.resolve()
    return args


if __name__ == "__main__":
    try:
        subprocess.run(build_command(parse_args()), cwd=ROOT, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)
