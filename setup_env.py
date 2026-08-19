import argparse
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from utils.i2s_format import validate_i2s_gguf


ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("celiums_bitnet.setup")

SUPPORTED_HF_MODELS = {
    "microsoft/BitNet-b1.58-2B-4T": "BitNet-b1.58-2B-4T",
}

ARCH_ALIAS = {
    "AMD64": "x86_64",
    "x86": "x86_64",
    "x86_64": "x86_64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "ARM64": "arm64",
}


def system_info():
    machine = platform.machine()
    if machine not in ARCH_ALIAS:
        raise RuntimeError(f"Unsupported architecture: {machine}")
    return platform.system(), ARCH_ALIAS[machine]


def run_command(command, log_path=None):
    LOGGER.info("Running: %s", " ".join(map(str, command)))
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as output:
            subprocess.run(command, cwd=ROOT, check=True, stdout=output, stderr=subprocess.STDOUT)
    else:
        subprocess.run(command, cwd=ROOT, check=True)


def resolve_model_dir(args):
    if args.hf_repo:
        return args.model_dir / SUPPORTED_HF_MODELS[args.hf_repo]
    return args.local_model_dir.resolve() if args.local_model_dir else args.model_dir


def download_model(args, model_dir):
    if not args.hf_repo:
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    command = ["huggingface-cli", "download", args.hf_repo, "--local-dir", str(model_dir)]
    if args.revision:
        command.extend(["--revision", args.revision])
    run_command(command, args.log_dir / "download_model.log")


def validate_source_model(model_dir):
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"Strict conversion requires {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    architectures = config.get("architectures", [])
    if not architectures or architectures[0] not in ("BitnetForCausalLM", "BitNetForCausalLM"):
        raise RuntimeError(f"Unsupported strict model architecture: {architectures}")
    if config.get("hidden_act") != "relu2":
        raise RuntimeError(f"Unsupported strict model activation: {config.get('hidden_act')}")


def generate_tl(args, model_dir, arch):
    if args.quant_type == "i2_s":
        return
    if args.use_pretuned:
        preset_dir = ROOT / "preset_kernels" / model_dir.name
        header = preset_dir / f"bitnet-lut-kernels-{args.quant_type}.h"
        config = preset_dir / f"kernel_config_{args.quant_type}.ini"
        if not header.exists():
            raise FileNotFoundError(f"Pretuned kernel not found: {header}")
        shutil.copyfile(header, ROOT / "include" / "bitnet-lut-kernels.h")
        if config.exists():
            shutil.copyfile(config, ROOT / "include" / "kernel_config.ini")
        return

    raise RuntimeError(
        "TL code generation remains experimental. Use --use-pretuned with a matching preset."
    )


def configure_and_build(args, arch):
    cmake = args.cmake or shutil.which("cmake")
    if not cmake:
        raise FileNotFoundError("cmake is required")

    c_compiler = args.c_compiler or shutil.which("clang") or shutil.which("gcc")
    cxx_compiler = args.cxx_compiler or shutil.which("clang++") or shutil.which("g++")
    if not c_compiler or not cxx_compiler:
        raise FileNotFoundError("A C and C++ compiler are required")

    configure = [
        cmake, "-S", str(ROOT), "-B", str(args.build_dir),
        f"-DCMAKE_BUILD_TYPE={args.build_type}",
        f"-DCMAKE_C_COMPILER={c_compiler}",
        f"-DCMAKE_CXX_COMPILER={cxx_compiler}",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_COMMON=ON",
        f"-DLLAMA_BUILD_EXAMPLES={'ON' if args.build_examples else 'OFF'}",
        f"-DCELIUMS_BITNET_BUILD_SERVER={'ON' if args.build_server else 'OFF'}",
        f"-DLLAMA_BUILD_TESTS={'ON' if args.build_tests else 'OFF'}",
        f"-DBITNET_ARM_TL1={'ON' if arch == 'arm64' and args.quant_type == 'tl1' else 'OFF'}",
        f"-DBITNET_X86_TL2={'ON' if arch == 'x86_64' and args.quant_type == 'tl2' else 'OFF'}",
    ]
    run_command(configure, args.log_dir / "configure.log")
    targets = ["llama-bench", "celiums-logits-capture"]
    if args.build_tests:
        targets.extend(["test-quantize-fns", "test-i2s-mul-mat", "test-celiums-hybrid"])
    if args.build_server:
        targets.extend(["llama-cli", "llama-server"])
    run_command(
        [cmake, "--build", str(args.build_dir), "--config", args.build_type,
         "--parallel", str(args.jobs), "--target", *targets],
        args.log_dir / "build.log",
    )


def convert_model(args, model_dir):
    if args.build_only:
        return
    if args.quant_embd:
        raise RuntimeError("--quant-embd is temporarily disabled pending strict embedding validation")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    output = model_dir / f"ggml-model-{args.quant_type}.gguf"
    if output.exists() and output.stat().st_size > 0 and not args.force_convert:
        validate_source_model(model_dir)
        if args.quant_type == "i2_s":
            validate_i2s_gguf(output)
        LOGGER.info("Using existing model: %s", output)
        return

    validate_source_model(model_dir)
    command = [
        sys.executable,
        str(ROOT / "utils" / "convert-hf-to-gguf-bitnet.py"),
        str(model_dir),
        "--outtype", args.quant_type,
        "--outfile", str(output),
    ]
    if args.quant_embd:
        command.append("--quant-embd")
    run_command(command, args.log_dir / "convert_model.log")
    if args.quant_type == "i2_s":
        validate_i2s_gguf(output)


def parse_args():
    _, arch = system_info()
    quant_choices = ["i2_s"]
    if arch == "arm64":
        quant_choices.append("tl1")
    if arch == "x86_64":
        quant_choices.append("tl2")

    parser = argparse.ArgumentParser(description="Build Celiums BitNet and optionally prepare a model")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--hf-repo", choices=SUPPORTED_HF_MODELS)
    source.add_argument("--local-model-dir", type=Path)
    parser.add_argument("--revision", help="Hugging Face revision to download")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--quant-type", "-q", choices=quant_choices, default="i2_s")
    parser.add_argument("--quant-embd", action="store_true")
    parser.add_argument("--use-pretuned", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--build-tests", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build-server", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build-examples", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--build-type", default="Release")
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--c-compiler")
    parser.add_argument("--cxx-compiler")
    parser.add_argument("--cmake", help="Path to the CMake executable")
    args = parser.parse_args()
    args.model_dir = args.model_dir.resolve()
    args.build_dir = args.build_dir.resolve()
    args.log_dir = args.log_dir.resolve()
    return args


def main():
    args = parse_args()
    _, arch = system_info()
    model_dir = resolve_model_dir(args)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    download_model(args, model_dir)
    generate_tl(args, model_dir, arch)
    configure_and_build(args, arch)
    convert_model(args, model_dir)


def signal_handler(_sig, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        LOGGER.error("%s", error)
        sys.exit(1)
