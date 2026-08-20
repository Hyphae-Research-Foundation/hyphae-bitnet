import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def git_value(path, *arguments, check=True):
    return subprocess.run(
        ["git", *arguments], cwd=path, check=check, capture_output=True, text=True
    ).stdout.strip()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Write a Celiums BitNet Runtime build manifest")
    parser.add_argument("--profile", choices=("native", "avx2", "scalar"), required=True)
    parser.add_argument("--build-type", default="Release")
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    submodule = ROOT / "3rdparty" / "llama.cpp"
    source_dirty = bool(git_value(ROOT, "status", "--porcelain", "--untracked-files=no"))
    if source_dirty and not args.allow_dirty:
        raise RuntimeError("Refusing to write a release manifest from a dirty source tree")
    manifest = {
        "schema": "celiums-bitnet-runtime-build-v1",
        "product_version": "0.2.0",
        "api_version": 1,
        "product_commit": git_value(ROOT, "rev-parse", "HEAD"),
        "engine_commit": git_value(submodule, "rev-parse", "HEAD"),
        "source_dirty": source_dirty,
        "profile": args.profile,
        "build_type": args.build_type,
        "compiler": args.compiler,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "artifacts": [],
    }
    if args.model:
        model = args.model.resolve()
        manifest["model"] = {
            "path": str(model),
            "size": model.stat().st_size,
            "sha256": sha256_file(model),
        }
    for artifact in args.artifact:
        path = artifact.resolve()
        manifest["artifacts"].append({
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
