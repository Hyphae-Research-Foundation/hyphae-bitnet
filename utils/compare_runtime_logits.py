import argparse
import ctypes
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    from .compare_logits import array_metrics, load_capture, paths_alias
except ImportError:
    from compare_logits import array_metrics, load_capture, paths_alias


STATUS_OK = 0
STATUS_BUFFER_TOO_SMALL = 3


class RuntimeOptions(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_size_t), ("api_version", ctypes.c_uint32)]


class ModelOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("api_version", ctypes.c_uint32),
        ("use_mmap", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
    ]


class SessionOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("api_version", ctypes.c_uint32),
        ("context_size", ctypes.c_uint32),
        ("batch_size", ctypes.c_uint32),
        ("ubatch_size", ctypes.c_uint32),
        ("threads", ctypes.c_int32),
        ("threads_batch", ctypes.c_int32),
    ]


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def decode_string(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def final_reference_row(tokens, positions, logits):
    final_position = len(tokens) - 1
    matches = np.flatnonzero(positions == final_position)
    if len(matches) != 1:
        raise ValueError(f"Reference must capture final prompt position {final_position}")
    return final_position, np.ascontiguousarray(logits[int(matches[0])])


def configure_library(path):
    library = ctypes.CDLL(str(path))
    handle = ctypes.c_void_p
    library.celiums_bitnet_runtime_default_options.restype = RuntimeOptions
    library.celiums_bitnet_model_default_options.restype = ModelOptions
    library.celiums_bitnet_session_default_options.restype = SessionOptions
    library.celiums_bitnet_runtime_create.argtypes = [ctypes.POINTER(RuntimeOptions), ctypes.POINTER(handle)]
    library.celiums_bitnet_runtime_create.restype = ctypes.c_int
    library.celiums_bitnet_runtime_destroy.argtypes = [handle]
    library.celiums_bitnet_model_load.argtypes = [handle, ctypes.c_char_p, ctypes.POINTER(ModelOptions), ctypes.POINTER(handle)]
    library.celiums_bitnet_model_load.restype = ctypes.c_int
    library.celiums_bitnet_model_destroy.argtypes = [handle]
    library.celiums_bitnet_session_create.argtypes = [handle, ctypes.POINTER(SessionOptions), ctypes.POINTER(handle)]
    library.celiums_bitnet_session_create.restype = ctypes.c_int
    library.celiums_bitnet_session_destroy.argtypes = [handle]
    library.celiums_bitnet_tokenize.argtypes = [
        handle, ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_size_t),
    ]
    library.celiums_bitnet_tokenize.restype = ctypes.c_int
    library.celiums_bitnet_session_prefill.argtypes = [
        handle, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.c_bool,
    ]
    library.celiums_bitnet_session_prefill.restype = ctypes.c_int
    library.celiums_bitnet_session_copy_logits.argtypes = [
        handle, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_size_t),
    ]
    library.celiums_bitnet_session_copy_logits.restype = ctypes.c_int
    for name in (
        "celiums_bitnet_version",
        "celiums_bitnet_product_commit",
        "celiums_bitnet_engine_commit",
        "celiums_bitnet_cpu_profile",
    ):
        getattr(library, name).restype = ctypes.c_char_p
    library.celiums_bitnet_status_string.argtypes = [ctypes.c_int]
    library.celiums_bitnet_status_string.restype = ctypes.c_char_p
    return library


def require_status(library, status, expected, operation):
    if status != expected:
        text = decode_string(library.celiums_bitnet_status_string(status))
        raise RuntimeError(f"{operation} failed: {text} ({status})")


def compare_runtime(library_path, model_path, reference_path):
    tokens, positions, logits, metadata, _ = load_capture(reference_path)
    final_position, reference = final_reference_row(tokens, positions, logits)
    if positions.size != 1:
        print(
            "warning: reference captures extra positions; the runtime prefill materializes only the final "
            "prompt position, so extra requested outputs can introduce ULP-level differences",
            file=sys.stderr,
        )
    prompt = decode_string(metadata["celiums.logits_capture.prompt"])
    if metadata["celiums.logits_capture.escape"]:
        raise ValueError("Escaped oracle prompts are not supported")

    library = configure_library(library_path)
    runtime = ctypes.c_void_p()
    model = ctypes.c_void_p()
    session = ctypes.c_void_p()
    try:
        runtime_options = library.celiums_bitnet_runtime_default_options()
        require_status(
            library,
            library.celiums_bitnet_runtime_create(ctypes.byref(runtime_options), ctypes.byref(runtime)),
            STATUS_OK,
            "runtime creation",
        )
        model_options = library.celiums_bitnet_model_default_options()
        require_status(
            library,
            library.celiums_bitnet_model_load(
                runtime, str(model_path).encode(), ctypes.byref(model_options), ctypes.byref(model)),
            STATUS_OK,
            "model load",
        )
        session_options = library.celiums_bitnet_session_default_options()
        session_options.context_size = metadata["celiums.logits_capture.n_ctx"]
        session_options.batch_size = metadata["celiums.logits_capture.n_batch"]
        session_options.ubatch_size = metadata["celiums.logits_capture.n_ubatch"]
        session_options.threads = metadata["celiums.logits_capture.n_threads"]
        session_options.threads_batch = metadata["celiums.logits_capture.n_threads_batch"]
        require_status(
            library,
            library.celiums_bitnet_session_create(model, ctypes.byref(session_options), ctypes.byref(session)),
            STATUS_OK,
            "session creation",
        )

        token_count = ctypes.c_size_t()
        prompt_bytes = prompt.encode("utf-8")
        add_special = bool(metadata["celiums.logits_capture.add_special"])
        parse_special = bool(metadata["celiums.logits_capture.parse_special"])
        require_status(
            library,
            library.celiums_bitnet_tokenize(
                model, prompt_bytes, add_special, parse_special, None, ctypes.byref(token_count)),
            STATUS_BUFFER_TOO_SMALL,
            "token count query",
        )
        token_buffer = (ctypes.c_int32 * token_count.value)()
        require_status(
            library,
            library.celiums_bitnet_tokenize(
                model, prompt_bytes, add_special, parse_special, token_buffer, ctypes.byref(token_count)),
            STATUS_OK,
            "tokenization",
        )
        if not np.array_equal(tokens, np.ctypeslib.as_array(token_buffer)):
            raise ValueError("Runtime token IDs differ from the oracle")

        require_status(
            library,
            library.celiums_bitnet_session_prefill(session, token_buffer, token_count.value, True),
            STATUS_OK,
            "prefill",
        )
        logits_count = ctypes.c_size_t()
        require_status(
            library,
            library.celiums_bitnet_session_copy_logits(session, None, ctypes.byref(logits_count)),
            STATUS_BUFFER_TOO_SMALL,
            "logits count query",
        )
        if logits_count.value != reference.size:
            raise ValueError(f"Runtime vocabulary differs: {logits_count.value} != {reference.size}")
        logits_buffer = (ctypes.c_float * logits_count.value)()
        require_status(
            library,
            library.celiums_bitnet_session_copy_logits(session, logits_buffer, ctypes.byref(logits_count)),
            STATUS_OK,
            "logits copy",
        )
        candidate = np.ctypeslib.as_array(logits_buffer).copy()
        return {
            "schema": "celiums-runtime-logits-comparison-v1",
            "library": str(library_path),
            "model": str(model_path),
            "reference": str(reference_path),
            "position": final_position,
            "shape": list(reference.shape),
            **array_metrics(reference, candidate),
            "top1_agreement": float(reference.argmax() == candidate.argmax()),
            "runtime_version": decode_string(library.celiums_bitnet_version()),
            "product_commit": decode_string(library.celiums_bitnet_product_commit()),
            "engine_commit": decode_string(library.celiums_bitnet_engine_commit()),
            "cpu_profile": decode_string(library.celiums_bitnet_cpu_profile()),
            "reference_build": decode_string(metadata["celiums.logits_capture.build.commit"]),
        }
    finally:
        if session.value:
            library.celiums_bitnet_session_destroy(session)
        if model.value:
            library.celiums_bitnet_model_destroy(model)
        if runtime.value:
            library.celiums_bitnet_runtime_destroy(runtime)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Runtime C API logits against a Celiums GGUF oracle")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--require-bitwise", action="store_true")
    parser.add_argument("--max-abs", type=float)
    parser.add_argument("--max-nmse", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for path in (args.library, args.model, args.reference):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    for name in ("max_abs", "max_nmse"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.expected_model_sha256 and len(args.expected_model_sha256) != 64:
        parser.error("--expected-model-sha256 must be a SHA-256 hex digest")
    if args.output and any(paths_alias(args.output, path) for path in (args.library, args.model, args.reference)):
        parser.error("--output must not alias an input")
    return args


def main():
    args = parse_args()
    if args.expected_model_sha256:
        actual = sha256_file(args.model)
        if actual.lower() != args.expected_model_sha256.lower():
            raise ValueError(f"Model SHA-256 differs: {actual}")
    report = compare_runtime(args.library.resolve(), args.model.resolve(), args.reference.resolve())
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    failed = (
        (args.require_bitwise and not report["bitwise_equal"])
        or (args.max_abs is not None and report["max_abs_error"] > args.max_abs)
        or (args.max_nmse is not None and (report["nmse"] is None or report["nmse"] > args.max_nmse))
    )
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Runtime logits comparison failed: {error}", file=sys.stderr)
        sys.exit(2)
