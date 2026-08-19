import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
GGUF_PY = ROOT / "3rdparty" / "llama.cpp" / "gguf-py"
sys.path.insert(0, str(GGUF_PY))

import gguf
from utils.i2s_format import quantize_to_i2_s, validate_i2s_gguf


def pack_i2s(weights, scale):
    values = np.ones(weights.size, dtype=np.uint8)
    flat = weights.reshape(-1)
    values[flat > 0.5] = 2
    values[flat < -0.5] = 0
    values = values.reshape(-1, 4, 32)
    packed = (
        (values[:, 0, :] << 6)
        | (values[:, 1, :] << 4)
        | (values[:, 2, :] << 2)
        | values[:, 3, :]
    ).reshape(-1)
    result = np.zeros(weights.size // 4 + 32, dtype=np.uint8)
    result[: len(packed)] = packed
    result[weights.size // 4 : weights.size // 4 + 4] = np.frombuffer(
        np.float32(scale).tobytes(), dtype=np.uint8
    )
    return result


class I2SConversionTests(unittest.TestCase):
    def test_production_packer_matches_oracle(self):
        shape = (2, 128)
        weights = np.resize(np.array([-1.0, 0.0, 1.0], dtype=np.float32), shape)
        np.testing.assert_array_equal(
            quantize_to_i2_s(weights, override_scale=0.25),
            pack_i2s(weights, 0.25),
        )

    def test_production_packer_rejects_invalid_shapes(self):
        for shape in ((64,), (1, 64), (2, 64), (3, 130)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    quantize_to_i2_s(np.ones(shape, dtype=np.float32))

    def test_writer_rejects_invalid_i2s_byte_count(self):
        writer = gguf.GGUFWriter(Path("unused.gguf"), "llama")
        with self.assertRaises(ValueError):
            writer.add_tensor(
                "blk.0.attn_q.weight",
                np.zeros(95, dtype=np.uint8),
                raw_shape=(2, 128),
                raw_dtype=gguf.GGMLQuantizationType.I2_S,
            )

    def test_writer_rejects_empty_i2s_tensor(self):
        writer = gguf.GGUFWriter(Path("unused.gguf"), "llama")
        with self.assertRaises(ValueError):
            writer.add_tensor(
                "blk.0.attn_q.weight",
                np.zeros(32, dtype=np.uint8),
                raw_shape=(0, 128),
                raw_dtype=gguf.GGMLQuantizationType.I2_S,
            )

    def test_writer_rejects_invalid_alignment(self):
        writer = gguf.GGUFWriter(Path("unused.gguf"), "llama")
        for alignment in (0, 3, 48):
            with self.subTest(alignment=alignment):
                with self.assertRaises(ValueError):
                    writer.add_custom_alignment(alignment)

    def test_i2s_fixture_metadata_and_layout(self):
        shape = (2, 128)
        weights = np.resize(np.array([-1.0, 0.0, 1.0], dtype=np.float32), shape)
        packed = pack_i2s(weights, 0.25)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "fixture.gguf"
            writer = gguf.GGUFWriter(output, "llama")
            writer.add_file_type(gguf.LlamaFileType.MOSTLY_I2_S)
            writer.add_tensor(
                "blk.0.attn_q.weight",
                packed,
                raw_shape=shape,
                raw_dtype=gguf.GGMLQuantizationType.I2_S,
            )
            writer.add_tensor("output_norm.weight", np.ones(128, dtype=np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            reader = gguf.GGUFReader(output)
            self.assertEqual(
                reader.fields["general.file_type"].contents(),
                gguf.LlamaFileType.MOSTLY_I2_S,
            )
            tensors = {tensor.name: tensor for tensor in reader.tensors}
            i2s = tensors["blk.0.attn_q.weight"]
            self.assertEqual(i2s.tensor_type, gguf.GGMLQuantizationType.I2_S)
            self.assertEqual(tuple(i2s.shape), (128, 2))
            self.assertEqual(i2s.n_bytes, np.prod(shape) // 4 + 32)
            self.assertEqual(len(i2s.data), np.prod(shape) // 4 + 32)
            self.assertAlmostEqual(
                np.frombuffer(i2s.data[np.prod(shape) // 4:].tobytes(), dtype=np.float32)[0],
                0.25,
            )
            self.assertEqual(
                tensors["output_norm.weight"].tensor_type,
                gguf.GGMLQuantizationType.F32,
            )
            validate_i2s_gguf(output, require_complete_model=False)


if __name__ == "__main__":
    unittest.main()
