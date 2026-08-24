import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()


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


def command_version(command):
    return subprocess.run(
        [command, "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0].strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Write a Hyphae BitNet Runtime build manifest")
    parser.add_argument("--profile", choices=("native", "avx2", "scalar"), required=True)
    parser.add_argument("--build-type", default="Release")
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--cxx-compiler", required=True)
    parser.add_argument("--rustc")
    parser.add_argument("--cargo-lock", type=Path)
    parser.add_argument("--hyphae-commit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def read_required_hash(path):
    value = path.read_text(encoding="utf-8").strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"Invalid provenance hash in {path.relative_to(ROOT)}")
    return value


def display_path(path):
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return path.name


def ignored_untracked_paths():
    ignored = {"stage-native/", "stage-avx2/", "stage-scalar/"}
    return ignored


def main():
    args = parse_args()
    status_lines = git_value(ROOT, "status", "--porcelain", "--untracked-files=all").splitlines()
    output_path = args.output.resolve()
    generated_paths = {artifact.resolve() for artifact in args.artifact}
    generated_paths.add(output_path)
    source_dirty = any(
        not (
            line.startswith("?? ") and (
                line[3:] in ignored_untracked_paths() or
                (ROOT / line[3:]).resolve() in generated_paths
            )
        )
        for line in status_lines
    )
    if source_dirty and not args.allow_dirty:
        raise RuntimeError("Refusing to write a release manifest from a dirty source tree")
    manifest = {
        "schema": "celiums-bitnet-runtime-build-v2",
        "product_version": VERSION,
        "api_version": 1,
        "product_commit": git_value(ROOT, "rev-parse", "HEAD"),
        "engine_commit": read_required_hash(ROOT / "3rdparty" / "llama.cpp" / "ENGINE_COMMIT"),
        "engine_tree": read_required_hash(ROOT / "cmake" / "ENGINE_TREE"),
        "source_dirty": source_dirty,
        "profile": args.profile,
        "build_type": args.build_type,
        "c_compiler": command_version(args.compiler),
        "cxx_compiler": command_version(args.cxx_compiler),
        "rustc": args.rustc,
        "cargo_lock_sha256": sha256_file(args.cargo_lock) if args.cargo_lock else None,
        "hyphae_commit": args.hyphae_commit,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "artifacts": [],
    }
    actual_engine_tree = subprocess.run(
        [str(ROOT / "scripts" / "compute-engine-tree.sh")], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if actual_engine_tree != manifest["engine_tree"]:
        raise RuntimeError("Vendored engine tree does not match cmake/ENGINE_TREE")
    if args.model:
        model = args.model.resolve()
        manifest["model"] = {
            "path": display_path(args.model),
            "size": model.stat().st_size,
            "sha256": sha256_file(model),
        }
    for artifact in args.artifact:
        path = artifact.resolve()
        manifest["artifacts"].append({
            "path": display_path(artifact),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
