import argparse
import platform
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def binary_path(name):
    candidates = [ROOT / "build" / "bin" / name]
    if platform.system() == "Windows":
        candidates.insert(0, ROOT / "build" / "bin" / "Release" / f"{name}.exe")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name} was not found; build Celiums BitNet first")


def build_command(args):
    threads_batch = args.threads_batch if args.threads_batch is not None else args.threads
    command = [
        str(binary_path("llama-cli")),
        "-m", str(args.model),
        "-n", str(args.n_predict),
        "-t", str(args.threads),
        "-tb", str(threads_batch),
        "-p", args.prompt,
        "-ngl", "0",
        "-c", str(args.ctx_size),
        "--temp", str(args.temperature),
    ]
    if args.cpu_mask:
        command.extend(["-C", args.cpu_mask, "--cpu-strict", "1"])
    elif args.hybrid_auto:
        command.append("--celiums-hybrid-auto")
    if args.conversation:
        command.append("-cnv")
    return command


def parse_args():
    parser = argparse.ArgumentParser(description="Run Celiums BitNet inference")
    parser.add_argument("-m", "--model", type=Path, required=True)
    parser.add_argument("-n", "--n-predict", type=int, default=128)
    parser.add_argument("-p", "--prompt", required=True)
    parser.add_argument("-t", "--threads", "--threads-decode", type=int, default=2)
    parser.add_argument("-tb", "--threads-batch", "--threads-prefill", type=int)
    parser.add_argument("-c", "--ctx-size", type=int, default=2048)
    parser.add_argument("-temp", "--temperature", type=float, default=0.8)
    parser.add_argument("--cpu-mask", help="CPU affinity mask passed to llama-cli")
    parser.add_argument("--hybrid-auto", action="store_true", help="Use Celiums phase-aware hybrid CPU defaults")
    parser.add_argument("-cnv", "--conversation", action="store_true")
    return parser.parse_args()


def signal_handler(_sig, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    try:
        subprocess.run(build_command(parse_args()), cwd=ROOT, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Inference failed: {error}", file=sys.stderr)
        sys.exit(1)
