import argparse
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def binary_path():
    candidates = [ROOT / "build" / "bin" / "celiums-bitnet"]
    if platform.system() == "Windows":
        candidates.insert(0, ROOT / "build" / "bin" / "Release" / "celiums-bitnet.exe")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    installed = shutil.which("celiums-bitnet")
    if installed:
        return Path(installed)
    raise FileNotFoundError("celiums-bitnet was not found; build or install Celiums BitNet Runtime first")


def build_command(args):
    threads_batch = args.threads_batch if args.threads_batch is not None else args.threads
    command = [
        str(binary_path()), "serve",
        "-m", str(args.model),
        "-c", str(args.ctx_size),
        "-t", str(args.threads),
        "-tb", str(threads_batch),
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.api_key_file:
        command.extend(["--api-key-file", str(args.api_key_file)])
    if args.allow_unauthenticated_remote:
        command.append("--allow-unauthenticated-remote")
    return command


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Celiums BitNet Runtime server")
    parser.add_argument("-m", "--model", type=Path, required=True)
    parser.add_argument("-t", "--threads", "--threads-single-token", type=int, default=2)
    parser.add_argument("-tb", "--threads-batch", type=int)
    parser.add_argument("-c", "--ctx-size", type=int, default=2048)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--allow-unauthenticated-remote", action="store_true")
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
