from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def quantize_to_i2_s(
    weights: np.ndarray,
    override_scale: float | None = None,
    already_quantized: bool = False,
    byteorder: str = "little",
) -> np.ndarray:
    if weights.ndim != 2:
        raise ValueError(f"I2_S requires a rank-2 tensor, got rank {weights.ndim}")

    rows, columns = weights.shape
    if rows <= 0 or columns <= 0 or columns % 128 != 0:
        raise ValueError(f"I2_S row width must be a positive multiple of 128, got {columns}")

    flat = weights.reshape(-1).astype(np.float32)
    if not np.all(np.isfinite(flat)):
        raise ValueError("I2_S weights must be finite")

    if override_scale is not None:
        scale = np.float32(override_scale)
        quantized = flat
    elif already_quantized:
        nonzero = np.abs(flat[np.abs(flat) > 1e-6])
        scale = np.float32(nonzero[0] if len(nonzero) else 1e-5)
        quantized = np.round(flat / scale).clip(-1, 1)
    else:
        scale = np.float32(max(float(np.mean(np.abs(flat))), 1e-5))
        quantized = np.round(flat / scale).clip(-1, 1)

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"I2_S scale must be finite and positive, got {scale}")

    codes = np.ones(flat.size, dtype=np.uint8)
    codes[quantized > 0.5] = 2
    codes[quantized < -0.5] = 0
    codes = codes.reshape(-1, 4, 32)
    packed = (
        (codes[:, 0, :] << 6)
        | (codes[:, 1, :] << 4)
        | (codes[:, 2, :] << 2)
        | codes[:, 3, :]
    ).reshape(-1)

    result = np.zeros(flat.size // 4 + 32, dtype=np.uint8)
    result[: len(packed)] = packed
    scale_dtype = np.dtype("<f4" if byteorder == "little" else ">f4")
    result[len(packed) : len(packed) + 4] = np.frombuffer(
        np.asarray(scale, dtype=scale_dtype).tobytes(), dtype=np.uint8
    )
    return result


def validate_i2s_gguf(output: Path, require_complete_model: bool = True) -> None:
    sys.path.insert(0, str(ROOT / "3rdparty" / "llama.cpp" / "gguf-py"))
    import gguf

    reader = gguf.GGUFReader(output)
    file_type = reader.fields.get("general.file_type")
    if file_type is None or file_type.contents() != gguf.LlamaFileType.MOSTLY_I2_S:
        actual = None if file_type is None else file_type.contents()
        raise RuntimeError(f"Invalid I2_S GGUF file type in {output}: {actual}")

    architecture = reader.fields.get("general.architecture")
    if require_complete_model and (architecture is None or architecture.contents() != "bitnet-b1.58"):
        actual = None if architecture is None else architecture.contents()
        raise RuntimeError(f"Unsupported strict I2_S architecture in {output}: {actual}")
    block_count_field = reader.fields.get("bitnet-b1.58.block_count")
    block_count = 0 if block_count_field is None else int(block_count_field.contents())
    if require_complete_model and block_count <= 0:
        raise RuntimeError(f"Invalid strict I2_S block count in {output}: {block_count}")

    projection_suffixes = (
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    )
    seen_projections = set()
    i2s_count = 0
    for tensor in reader.tensors:
        matching_suffix = next((suffix for suffix in projection_suffixes if tensor.name.endswith(suffix)), None)
        if tensor.tensor_type == gguf.GGMLQuantizationType.I2_S:
            if len(tensor.shape) != 2 or any(int(dim) <= 0 for dim in tensor.shape) or int(tensor.shape[0]) % 128 != 0:
                raise RuntimeError(f"Invalid I2_S tensor shape for {tensor.name}: {tuple(tensor.shape)}")
            expected_bytes = tensor.n_elements // 4 + 32
            if tensor.n_bytes != expected_bytes or len(tensor.data) != expected_bytes:
                raise RuntimeError(
                    f"Invalid I2_S byte count for {tensor.name}: "
                    f"expected {expected_bytes}, got {tensor.n_bytes}/{len(tensor.data)}"
                )
            scale_dtype = np.dtype("<f4" if reader.endianess == gguf.GGUFEndian.LITTLE else ">f4")
            scale = np.frombuffer(
                tensor.data[tensor.n_elements // 4 : tensor.n_elements // 4 + 4].tobytes(),
                dtype=scale_dtype,
            )[0]
            if not np.isfinite(scale) or scale <= 0:
                raise RuntimeError(f"Invalid I2_S scale for {tensor.name}: {scale}")
            i2s_count += 1
            if matching_suffix:
                seen_projections.add(matching_suffix)
        elif matching_suffix:
            raise RuntimeError(f"Eligible projection is not I2_S: {tensor.name}")

    if i2s_count == 0:
        raise RuntimeError(f"I2_S GGUF contains no I2_S tensors: {output}")
    if require_complete_model:
        missing = sorted(set(projection_suffixes) - seen_projections)
        if missing:
            raise RuntimeError(f"I2_S GGUF is missing required projections: {', '.join(missing)}")
        names = {tensor.name for tensor in reader.tensors}
        missing_by_block = [
            f"blk.{block}.{suffix}"
            for block in range(block_count)
            for suffix in projection_suffixes
            if f"blk.{block}.{suffix}" not in names
        ]
        if missing_by_block:
            preview = ", ".join(missing_by_block[:8])
            suffix = "..." if len(missing_by_block) > 8 else ""
            raise RuntimeError(f"I2_S GGUF is missing block projections: {preview}{suffix}")
