import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
GGUF_PY = ROOT / "3rdparty" / "llama.cpp" / "gguf-py"
sys.path.insert(0, str(GGUF_PY))

import gguf


CORE_TENSORS = {"tokens", "positions", "logits"}


def paths_alias(left, right):
    left = left.resolve()
    right = right.resolve()
    if left == right:
        return True
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def field_contents(reader, key):
    field = reader.fields.get(key)
    if field is None:
        raise ValueError(f"Capture metadata is missing: {key}")
    return field.contents()


def load_capture(path):
    reader = gguf.GGUFReader(path)
    if field_contents(reader, "general.type") != "celiums-logits-capture":
        raise ValueError(f"Not a Celiums logits capture: {path}")
    version = field_contents(reader, "celiums.logits_capture.version")
    if version not in (1, 2):
        raise ValueError(f"Unsupported capture version: {path}")

    tensors = {tensor.name: tensor for tensor in reader.tensors}
    missing = sorted(CORE_TENSORS - tensors.keys())
    unexpected = sorted(
        name for name in tensors if name not in CORE_TENSORS and not name.startswith("probe.")
    )
    if missing or unexpected:
        raise ValueError(f"Invalid capture tensors; missing={missing}, unexpected={unexpected}")

    tokens_tensor = tensors["tokens"]
    positions_tensor = tensors["positions"]
    logits_tensor = tensors["logits"]
    if tokens_tensor.tensor_type != gguf.GGMLQuantizationType.I32 or len(tokens_tensor.shape) != 1:
        raise ValueError("tokens must be a rank-1 I32 tensor")
    if positions_tensor.tensor_type != gguf.GGMLQuantizationType.I32 or len(positions_tensor.shape) != 1:
        raise ValueError("positions must be a rank-1 I32 tensor")
    if logits_tensor.tensor_type != gguf.GGMLQuantizationType.F32 or len(logits_tensor.shape) not in (1, 2):
        raise ValueError("logits must be a rank-1 or rank-2 F32 tensor")

    tokens = np.asarray(tokens_tensor.data).copy()
    positions = np.asarray(positions_tensor.data).copy()
    logits = np.asarray(logits_tensor.data).copy()
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    if tokens.size == 0 or positions.size == 0 or logits.size == 0:
        raise ValueError("capture tensors must be nonempty")
    if logits.ndim != 2 or logits.shape[0] != len(positions):
        raise ValueError(f"Invalid logits shape: {logits.shape}")

    metadata = {
        key: field.contents()
        for key, field in reader.fields.items()
        if key.startswith("celiums.logits_capture.")
    }
    required_metadata = {
        "celiums.logits_capture.version",
        "celiums.logits_capture.model.path",
        "celiums.logits_capture.prompt",
        "celiums.logits_capture.add_special",
        "celiums.logits_capture.parse_special",
        "celiums.logits_capture.escape",
        "celiums.logits_capture.n_vocab",
        "celiums.logits_capture.n_tokens",
        "celiums.logits_capture.n_positions",
        "celiums.logits_capture.n_ctx",
        "celiums.logits_capture.n_batch",
        "celiums.logits_capture.n_ubatch",
        "celiums.logits_capture.n_threads",
        "celiums.logits_capture.n_threads_batch",
        "celiums.logits_capture.model.size",
        "celiums.logits_capture.model.parameters",
        "celiums.logits_capture.build.commit",
        "celiums.logits_capture.build.number",
        "celiums.logits_capture.build.compiler",
        "celiums.logits_capture.build.target",
        "celiums.logits_capture.system_info",
    }
    missing_metadata = required_metadata - metadata.keys()
    if missing_metadata:
        raise ValueError(f"Capture metadata is missing: {sorted(missing_metadata)}")
    counts = {
        "celiums.logits_capture.n_tokens": len(tokens),
        "celiums.logits_capture.n_positions": len(positions),
        "celiums.logits_capture.n_vocab": logits.shape[1],
    }
    for key, expected in counts.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Capture metadata mismatch for {key}")
    if np.any(positions < 0) or np.any(positions >= len(tokens)):
        raise ValueError("Capture position is outside the token tensor")
    if not np.all(positions[:-1] < positions[1:]):
        raise ValueError("Capture positions must be sorted and unique")

    probe_names = sorted(name for name in tensors if name.startswith("probe."))
    if version == 1 and probe_names:
        raise ValueError("Version 1 captures cannot contain probes")
    if version == 2 and metadata.get("celiums.logits_capture.n_probes") != len(probe_names):
        raise ValueError("Probe count metadata does not match probe tensors")
    probes = []
    for tensor_name in probe_names:
        tensor = tensors[tensor_name]
        prefix = f"celiums.logits_capture.{tensor_name}"
        probe = {
            "tensor_name": tensor_name,
            "name": metadata.get(prefix + ".name"),
            "op": metadata.get(prefix + ".op"),
            "type": metadata.get(prefix + ".type"),
            "decode_start": metadata.get(prefix + ".decode_start"),
            "decode_tokens": metadata.get(prefix + ".decode_tokens"),
            "tensor_type": tensor.tensor_type,
            "shape": tuple(int(value) for value in tensor.shape),
            "data": np.asarray(tensor.data).copy(),
        }
        required_probe = ("name", "op", "type", "decode_start", "decode_tokens")
        if any(probe[key] is None for key in required_probe):
            raise ValueError(f"Probe metadata is incomplete: {tensor_name}")
        if probe["type"] != tensor.tensor_type.name.lower():
            raise ValueError(f"Probe type metadata mismatch: {tensor_name}")
        probes.append(probe)
    return tokens, positions, logits, metadata, probes


def array_metrics(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError(f"Tensor shapes differ: {reference.shape} != {candidate.shape}")
    if reference.dtype != candidate.dtype:
        raise ValueError(f"Tensor dtypes differ: {reference.dtype} != {candidate.dtype}")
    if np.issubdtype(reference.dtype, np.floating):
        if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
            raise ValueError("Capture contains non-finite floating-point values")
        unsigned_dtype = np.dtype(f"u{reference.dtype.itemsize}")
        reference_bits = reference.view(unsigned_dtype)
        candidate_bits = candidate.view(unsigned_dtype)
        delta = candidate.astype(np.float64) - reference.astype(np.float64)
        abs_delta = np.abs(delta)
        mse = float(np.square(delta).mean())
        denominator = float(np.square(reference.astype(np.float64)).mean())
        nmse = 0.0 if mse == 0.0 else (math.inf if denominator == 0.0 else mse / denominator)
        return {
            "bitwise_equal": bool(np.array_equal(reference_bits, candidate_bits)),
            "differing_values": int(np.count_nonzero(reference_bits != candidate_bits)),
            "max_abs_error": float(abs_delta.max(initial=0.0)),
            "mean_abs_error": float(abs_delta.mean()),
            "mse": mse,
            "nmse": None if not math.isfinite(nmse) else nmse,
        }
    return {
        "bitwise_equal": bool(np.array_equal(reference, candidate)),
        "differing_values": int(np.count_nonzero(reference != candidate)),
        "max_abs_error": None,
        "mean_abs_error": None,
        "mse": None,
        "nmse": None,
    }


def compare(reference_path, candidate_path, compare_probes=True):
    ref_tokens, ref_positions, reference, ref_metadata, ref_probes = load_capture(reference_path)
    cand_tokens, cand_positions, candidate, cand_metadata, cand_probes = load_capture(candidate_path)
    if not np.array_equal(ref_tokens, cand_tokens):
        raise ValueError("Token IDs differ")
    if not np.array_equal(ref_positions, cand_positions):
        raise ValueError("Capture positions differ")
    identity_keys = (
        "celiums.logits_capture.n_ctx",
        "celiums.logits_capture.n_batch",
        "celiums.logits_capture.n_ubatch",
        "celiums.logits_capture.n_threads",
        "celiums.logits_capture.n_threads_batch",
        "celiums.logits_capture.model.size",
        "celiums.logits_capture.model.parameters",
    )
    for key in identity_keys:
        if ref_metadata.get(key) != cand_metadata.get(key):
            raise ValueError(f"Capture metadata differs for {key}")

    logits_metrics = array_metrics(reference, candidate)
    top1_reference = reference.argmax(axis=1)
    top1_candidate = candidate.argmax(axis=1)
    report = {
        "schema": "celiums-logits-comparison-v2",
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "shape": list(reference.shape),
        "positions": ref_positions.tolist(),
        **logits_metrics,
        "top1_agreement": float(np.mean(top1_reference == top1_candidate)),
        "reference_build": ref_metadata.get("celiums.logits_capture.build.commit"),
        "candidate_build": cand_metadata.get("celiums.logits_capture.build.commit"),
        "probes": [],
    }

    if not compare_probes:
        return report
    if len(ref_probes) != len(cand_probes):
        raise ValueError("Probe counts differ")
    for reference_probe, candidate_probe in zip(ref_probes, cand_probes):
        identity = ("name", "op", "type", "decode_start", "decode_tokens", "shape")
        if any(reference_probe[key] != candidate_probe[key] for key in identity):
            raise ValueError(
                f"Probe identity differs: {reference_probe['tensor_name']} != {candidate_probe['tensor_name']}"
            )
        metrics = array_metrics(reference_probe["data"], candidate_probe["data"])
        report["probes"].append({
            "tensor_name": reference_probe["tensor_name"],
            "name": reference_probe["name"],
            "op": reference_probe["op"],
            "type": reference_probe["type"],
            "decode_start": reference_probe["decode_start"],
            "decode_tokens": reference_probe["decode_tokens"],
            "shape": list(reference_probe["shape"]),
            **metrics,
        })
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Celiums BitNet logits and probe captures")
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--require-bitwise", action="store_true")
    parser.add_argument("--require-probes-bitwise", action="store_true")
    parser.add_argument("--ignore-probes", action="store_true")
    parser.add_argument("--max-abs", type=float)
    parser.add_argument("--max-nmse", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in ("max_abs", "max_nmse"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.output and (
        paths_alias(args.output, args.reference) or paths_alias(args.output, args.candidate)
    ):
        parser.error("--output must not alias either capture")
    if args.ignore_probes and args.require_probes_bitwise:
        parser.error("--ignore-probes cannot be combined with --require-probes-bitwise")
    return args


def main():
    args = parse_args()
    report = compare(args.reference, args.candidate, compare_probes=not args.ignore_probes)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    failed = (
        (args.require_bitwise and not report["bitwise_equal"])
        or (args.require_probes_bitwise and any(not probe["bitwise_equal"] for probe in report["probes"]))
        or (args.max_abs is not None and report["max_abs_error"] > args.max_abs)
        or (args.max_nmse is not None and (report["nmse"] is None or report["nmse"] > args.max_nmse))
    )
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"Logits comparison failed: {error}", file=sys.stderr)
        sys.exit(2)
