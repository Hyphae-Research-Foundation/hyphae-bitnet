import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
GGUF_PY = ROOT / "3rdparty" / "llama.cpp" / "gguf-py"
sys.path.insert(0, str(GGUF_PY))

import gguf
from utils.compare_logits import compare, load_capture
from utils.compare_runtime_logits import final_reference_row


def write_capture(
    path,
    logits,
    tokens=(1, 2, 3),
    positions=(2,),
    commit="test",
    tokens_dtype=np.int32,
    logits_dtype=np.float32,
    version=1,
    probes=(),
):
    writer = gguf.GGUFWriter(path, "celiums-logits-capture")
    writer.add_string("general.type", "celiums-logits-capture")
    writer.add_uint32("celiums.logits_capture.version", version)
    writer.add_string("celiums.logits_capture.model.path", "model.gguf")
    writer.add_string("celiums.logits_capture.prompt", "prompt")
    writer.add_bool("celiums.logits_capture.add_special", True)
    writer.add_bool("celiums.logits_capture.parse_special", False)
    writer.add_bool("celiums.logits_capture.escape", False)
    writer.add_string("celiums.logits_capture.build.commit", commit)
    writer.add_uint32("celiums.logits_capture.build.number", 1)
    writer.add_string("celiums.logits_capture.build.compiler", "test")
    writer.add_string("celiums.logits_capture.build.target", "test")
    writer.add_string("celiums.logits_capture.system_info", "test")
    writer.add_uint32("celiums.logits_capture.n_tokens", len(tokens))
    writer.add_uint32("celiums.logits_capture.n_positions", len(positions))
    writer.add_uint32("celiums.logits_capture.n_vocab", len(logits[0]))
    writer.add_uint32("celiums.logits_capture.n_ctx", 32)
    writer.add_uint32("celiums.logits_capture.n_batch", 8)
    writer.add_uint32("celiums.logits_capture.n_ubatch", 8)
    writer.add_uint32("celiums.logits_capture.n_threads", 1)
    writer.add_uint32("celiums.logits_capture.n_threads_batch", 1)
    writer.add_uint64("celiums.logits_capture.model.size", 1)
    writer.add_uint64("celiums.logits_capture.model.parameters", 1)
    if version == 2:
        writer.add_uint32("celiums.logits_capture.n_probes", len(probes))
    writer.add_tensor("tokens", np.asarray(tokens, dtype=tokens_dtype))
    writer.add_tensor("positions", np.asarray(positions, dtype=np.int32))
    writer.add_tensor("logits", np.asarray(logits, dtype=logits_dtype))
    for index, probe in enumerate(probes):
        tensor_name = f"probe.{index:04d}"
        data = np.asarray(probe["data"], dtype=np.float32)
        prefix = f"celiums.logits_capture.{tensor_name}"
        writer.add_string(prefix + ".name", probe["name"])
        writer.add_string(prefix + ".op", probe["op"])
        writer.add_string(prefix + ".type", "f32")
        writer.add_int32(prefix + ".decode_start", 0)
        writer.add_int32(prefix + ".decode_tokens", len(tokens))
        writer.add_tensor(tensor_name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class LogitsComparisonTests(unittest.TestCase):
    def test_runtime_reference_selects_final_prompt_position(self):
        tokens = np.array([10, 11, 12], dtype=np.int32)
        positions = np.array([0, 2], dtype=np.int32)
        logits = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        position, row = final_reference_row(tokens, positions, logits)
        self.assertEqual(position, 2)
        np.testing.assert_array_equal(row, logits[1])

    def test_runtime_reference_requires_final_prompt_position(self):
        tokens = np.array([10, 11, 12], dtype=np.int32)
        positions = np.array([0], dtype=np.int32)
        logits = np.array([[1.0, 2.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "final prompt position 2"):
            final_reference_row(tokens, positions, logits)

    def test_identical_capture_is_bitwise_equal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "capture.gguf"
            write_capture(path, [[1.0, 2.0, 3.0]])
            report = compare(path, path)
            self.assertTrue(report["bitwise_equal"])
            self.assertEqual(report["differing_values"], 0)
            self.assertEqual(report["nmse"], 0.0)

    def test_comparison_reports_numeric_differences(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reference = Path(temp_dir) / "reference.gguf"
            candidate = Path(temp_dir) / "candidate.gguf"
            write_capture(reference, [[1.0, 2.0, 3.0]])
            write_capture(candidate, [[1.0, 2.5, 3.0]])
            report = compare(reference, candidate)
            self.assertFalse(report["bitwise_equal"])
            self.assertEqual(report["differing_values"], 1)
            self.assertEqual(report["max_abs_error"], 0.5)
            self.assertEqual(report["top1_agreement"], 1.0)

    def test_differing_values_counts_float_elements(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reference = Path(temp_dir) / "reference.gguf"
            candidate = Path(temp_dir) / "candidate.gguf"
            write_capture(reference, [[0.0, 1.0]])
            write_capture(candidate, [[-0.0, 1.0]])
            report = compare(reference, candidate)
            self.assertFalse(report["bitwise_equal"])
            self.assertEqual(report["differing_values"], 1)

    def test_probe_comparison_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reference = Path(temp_dir) / "reference.gguf"
            candidate = Path(temp_dir) / "candidate.gguf"
            write_capture(reference, [[1.0, 2.0]], version=2)
            write_capture(
                candidate,
                [[1.0, 2.0]],
                version=2,
                probes=({"name": "probe", "op": "MUL", "data": [1.0, 2.0]},),
            )
            with self.assertRaisesRegex(ValueError, "Probe counts differ"):
                compare(reference, candidate)
            report = compare(reference, candidate, compare_probes=False)
            self.assertTrue(report["bitwise_equal"])
            self.assertEqual(report["probes"], [])

    def test_mismatched_tokens_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reference = Path(temp_dir) / "reference.gguf"
            candidate = Path(temp_dir) / "candidate.gguf"
            write_capture(reference, [[1.0, 2.0]], tokens=(1, 2), positions=(1,))
            write_capture(candidate, [[1.0, 2.0]], tokens=(1, 3), positions=(1,))
            with self.assertRaisesRegex(ValueError, "Token IDs differ"):
                compare(reference, candidate)

    def test_capture_shape_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "capture.gguf"
            write_capture(path, [[1.0, 2.0], [3.0, 4.0]], positions=(0, 2))
            tokens, positions, logits, metadata, probes = load_capture(path)
            self.assertEqual(tokens.tolist(), [1, 2, 3])
            self.assertEqual(positions.tolist(), [0, 2])
            self.assertEqual(logits.shape, (2, 2))
            self.assertEqual(metadata["celiums.logits_capture.build.commit"], "test")
            self.assertEqual(probes, [])

    def test_wrong_tensor_types_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wrong_tokens = Path(temp_dir) / "wrong-tokens.gguf"
            wrong_logits = Path(temp_dir) / "wrong-logits.gguf"
            write_capture(wrong_tokens, [[1.0, 2.0]], tokens_dtype=np.float32)
            write_capture(wrong_logits, [[1.0, 2.0]], logits_dtype=np.float16)
            with self.assertRaisesRegex(ValueError, "tokens must be"):
                load_capture(wrong_tokens)
            with self.assertRaisesRegex(ValueError, "logits must be"):
                load_capture(wrong_logits)

    def test_out_of_range_positions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "capture.gguf"
            write_capture(path, [[1.0, 2.0]], positions=(3,))
            with self.assertRaisesRegex(ValueError, "outside the token tensor"):
                load_capture(path)


if __name__ == "__main__":
    unittest.main()
