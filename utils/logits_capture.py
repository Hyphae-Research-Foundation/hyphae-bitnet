import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ENGINE_COMMIT = ROOT / "3rdparty" / "llama.cpp" / "ENGINE_COMMIT"
ENGINE_TREE = ROOT / "cmake" / "ENGINE_TREE"


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def paths_alias(left, right):
    left = left.resolve()
    right = right.resolve()
    if left == right:
        return True
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def capture_binary(build_dir):
    candidates = [build_dir / "bin" / "celiums-logits-capture"]
    if platform.system() == "Windows":
        candidates.insert(0, build_dir / "bin" / "Release" / "celiums-logits-capture.exe")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("celiums-logits-capture was not found; build Celiums BitNet first")


def build_command(args):
    threads_batch = args.threads_batch if args.threads_batch is not None else args.threads
    command = [
        str(capture_binary(args.build_dir)),
        "--model", str(args.model),
        "--prompt", args.prompt,
        "--output", str(args.output),
        "--ctx-size", str(args.ctx_size),
        "--batch-size", str(args.batch_size),
        "--ubatch-size", str(args.ubatch_size),
        "--threads", str(args.threads),
        "--threads-batch", str(threads_batch),
        "--fit", "off",
        "--gpu-layers", "0",
    ]
    for position in args.position:
        command.append(f"--capture-position={position}")
    for tensor in args.tensor:
        command.append(f"--capture-tensor={tensor}")
    if args.cpu_mask:
        command.extend([
            "--cpu-mask", args.cpu_mask, "--cpu-strict", "1",
            "--cpu-mask-batch", args.cpu_mask, "--cpu-strict-batch", "1",
        ])
    return command


def parse_args():
    parser = argparse.ArgumentParser(description="Capture deterministic Celiums BitNet logits")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--position", type=int, action="append", default=[])
    parser.add_argument("--tensor", action="append", default=[], help="Capture NAME or NAME@OP")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--threads-batch", type=int)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--cpu-mask")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    args.build_dir = args.build_dir.resolve()
    if args.metadata_output is None:
        args.metadata_output = args.output.with_suffix(args.output.suffix + ".json")
    else:
        args.metadata_output = args.metadata_output.resolve()
    for left, right in (
        (args.model, args.output),
        (args.model, args.metadata_output),
        (args.output, args.metadata_output),
    ):
        if paths_alias(left, right):
            parser.error(f"Paths must be distinct: {left} and {right}")
    return args


def main():
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(args)
    binary = Path(command[0]).resolve()
    subprocess.run(command, cwd=ROOT, check=True)
    identity = subprocess.run(
        [str(binary), "--version"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    root_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    root_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    product_commit = root_commit[:9]
    engine_commit = ENGINE_COMMIT.read_text(encoding="utf-8").strip()
    engine_tree = ENGINE_TREE.read_text(encoding="utf-8").strip()
    for expected in (product_commit, engine_commit, engine_tree):
        if expected not in identity:
            raise RuntimeError("Capture binary provenance does not match the current source tree")
    metadata = {
        "schema": "celiums-logits-capture-run-v2",
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "capture": str(args.output),
        "capture_sha256": sha256_file(args.output),
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "product_commit": root_commit,
        "engine_commit": engine_commit,
        "engine_tree": engine_tree,
        "source_dirty": bool(root_status),
        "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "requested_positions": args.position or [-1],
        "requested_tensors": args.tensor,
        "command": command,
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Logits capture failed: {error}", file=sys.stderr)
        sys.exit(1)
