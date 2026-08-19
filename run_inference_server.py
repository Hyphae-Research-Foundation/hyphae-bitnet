import argparse
import ipaddress
import platform
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def binary_path():
    candidates = [ROOT / "build" / "bin" / "llama-server"]
    if platform.system() == "Windows":
        candidates.insert(0, ROOT / "build" / "bin" / "Release" / "llama-server.exe")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("llama-server was not found; build Celiums BitNet first")


def build_command(args):
    threads_batch = args.threads_batch if args.threads_batch is not None else args.threads
    command = [
        str(binary_path()),
        "-m", str(args.model),
        "-c", str(args.ctx_size),
        "-t", str(args.threads),
        "-tb", str(threads_batch),
        "-n", str(args.n_predict),
        "-ngl", "0",
        "--temp", str(args.temperature),
        "--host", args.host,
        "--port", str(args.port),
        "-cb",
    ]
    if args.cpu_mask:
        command.extend(["-C", args.cpu_mask, "--cpu-strict", "1"])
    elif args.hybrid_auto:
        command.append("--celiums-hybrid-auto")
    if args.prompt:
        command.extend(["-p", args.prompt])
    return command


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Celiums BitNet llama.cpp server wrapper")
    parser.add_argument("-m", "--model", type=Path, required=True)
    parser.add_argument("-p", "--prompt")
    parser.add_argument("-n", "--n-predict", type=int, default=4096)
    parser.add_argument("-t", "--threads", "--threads-single-token", type=int, default=2)
    parser.add_argument("-tb", "--threads-batch", type=int)
    parser.add_argument("-c", "--ctx-size", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--cpu-mask")
    parser.add_argument("--hybrid-auto", action="store_true", help="Use Celiums phase-aware hybrid CPU defaults")
    return parser.parse_args()


def signal_handler(_sig, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args()
    try:
        try:
            is_loopback = ipaddress.ip_address(args.host).is_loopback
        except ValueError:
            is_loopback = args.host.lower() == "localhost"
        if not is_loopback:
            print("Warning: llama-server has no authentication configured by this wrapper.", file=sys.stderr)
        subprocess.run(build_command(args), cwd=ROOT, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Server failed: {error}", file=sys.stderr)
        sys.exit(1)
