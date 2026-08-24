import argparse
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def binary_path():
    candidates = [ROOT / "build" / "bin" / name for name in ("hyphae-bitnet", "celiums-bitnet")]
    if platform.system() == "Windows":
        candidates.insert(0, ROOT / "build" / "bin" / "Release" / "hyphae-bitnet.exe")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    installed = shutil.which("hyphae-bitnet") or shutil.which("celiums-bitnet")
    if installed:
        return Path(installed)
    raise FileNotFoundError("hyphae-bitnet was not found; build or install Hyphae BitNet Runtime first")


def build_command(args):
    threads_batch = args.threads_batch if args.threads_batch is not None else args.threads
    command = [
        str(binary_path()), "serve",
        "-m", str(args.model),
        "--model-family", getattr(args, "model_family", "bitnet"),
        "-c", str(args.ctx_size),
        "-t", str(args.threads),
        "-tb", str(threads_batch),
        "--host", args.host,
        "--port", str(args.port),
    ]
    if getattr(args, "n_seq", None):
        command.extend(["--n-seq", str(args.n_seq)])
    if getattr(args, "ram_budget_bytes", None):
        command.extend(["--ram-budget-bytes", str(args.ram_budget_bytes)])
    if getattr(args, "compute_layout", None) is not None:
        command.extend(["--compute-layout", "1" if args.compute_layout else "0"])
    if args.api_key_file:
        command.extend(["--api-key-file", str(args.api_key_file)])
    if args.allow_unauthenticated_remote:
        command.append("--allow-unauthenticated-remote")
    return command


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Hyphae BitNet Runtime server")
    parser.add_argument("-m", "--model", type=Path, required=True)
    parser.add_argument("--model-family", choices=["bitnet", "bonsai"], default="bitnet")
    parser.add_argument("-t", "--threads", "--threads-single-token", type=int, default=2)
    parser.add_argument("-tb", "--threads-batch", type=int)
    parser.add_argument("-c", "--ctx-size", type=int, default=2048)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--allow-unauthenticated-remote", action="store_true")
    parser.add_argument("--n-seq", type=int, default=1)
    parser.add_argument("--ram-budget-bytes", type=int, default=0)
    parser.add_argument("--compute-layout", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def signal_handler(_sig, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    try:
        subprocess.run(build_command(parse_args()), cwd=ROOT, check=True)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Server failed: {error}", file=sys.stderr)
        sys.exit(1)
