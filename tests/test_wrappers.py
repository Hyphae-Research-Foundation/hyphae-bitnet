import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inference = load_module("celiums_run_inference", ROOT / "run_inference.py")
server = load_module("celiums_run_server", ROOT / "run_inference_server.py")
benchmark = load_module("celiums_benchmark", ROOT / "utils" / "e2e_benchmark.py")
logits_capture = load_module("celiums_logits_capture", ROOT / "utils" / "logits_capture.py")


class WrapperCommandTests(unittest.TestCase):
    def test_inference_thread_defaults(self):
        args = SimpleNamespace(
            model=Path("model.gguf"), n_predict=32, threads=4, threads_batch=None,
            prompt="hello world", ctx_size=512, temperature=0.0,
        )
        with patch.object(inference, "binary_path", return_value=Path("celiums-bitnet")):
            command = inference.build_command(args)
        self.assertEqual(command[1], "run")
        self.assertEqual(command[command.index("-t") + 1], "4")
        self.assertEqual(command[command.index("-tb") + 1], "4")
        self.assertIn("hello world", command)

    def test_inference_separate_threads(self):
        args = SimpleNamespace(
            model=Path("model.gguf"), n_predict=32, threads=6, threads_batch=14,
            prompt="prompt", ctx_size=512, temperature=0.8,
        )
        with patch.object(inference, "binary_path", return_value=Path("celiums-bitnet")):
            command = inference.build_command(args)
        self.assertEqual(command[command.index("-tb") + 1], "14")
        self.assertNotIn("--cpu-strict", command)

    def test_server_uses_separate_thread_counts(self):
        args = SimpleNamespace(
            model=Path("model.gguf"), ctx_size=1024, threads=4, threads_batch=12,
            host="127.0.0.1", port=8080,
            api_key_file=None, allow_unauthenticated_remote=False,
        )
        with patch.object(server, "binary_path", return_value=Path("celiums-bitnet")):
            command = server.build_command(args)
        self.assertEqual(command[1], "serve")
        self.assertEqual(command[command.index("-tb") + 1], "12")

    def test_server_wrapper_defers_bind_policy_to_native_server(self):
        args = SimpleNamespace(
            model=Path("model.gguf"), ctx_size=1024, threads=4, threads_batch=12,
            host="0.0.0.0", port=8080, api_key_file=None,
            allow_unauthenticated_remote=False,
        )
        with patch.object(server, "binary_path", return_value=Path("celiums-bitnet")):
            command = server.build_command(args)
        self.assertEqual(command[command.index("--host") + 1], "0.0.0.0")

    def test_benchmark_uses_real_batch(self):
        args = SimpleNamespace(
            build_dir=ROOT / "build", model=Path("model.gguf"), n_prompt=256,
            n_token=128, batch=256, ubatch=64, threads=8, repetitions=5,
            output="jsonl", cpu_mask=None,
        )
        with patch.object(benchmark, "benchmark_binary", return_value=Path("celiums-bitnet")):
            command = benchmark.build_command(args)
        self.assertEqual(command[1], "bench")
        self.assertEqual(command[command.index("-b") + 1], "256")
        self.assertEqual(command[command.index("-ub") + 1], "64")
        self.assertNotEqual(command[command.index("-b") + 1], "1")

    def test_native_benchmark_rejects_cpu_mask(self):
        args = SimpleNamespace(
            build_dir=ROOT / "build", model=Path("model.gguf"), n_prompt=128,
            n_token=128, batch=128, ubatch=64, threads=8, repetitions=5,
            output="jsonl", cpu_mask="0xff",
        )
        with patch.object(benchmark, "benchmark_binary", return_value=Path("celiums-bitnet")):
            with self.assertRaisesRegex(ValueError, "CPU masks"):
                benchmark.build_command(args)

    def test_setup_lists_only_strictly_supported_model(self):
        setup = load_module("celiums_setup", ROOT / "setup_env.py")
        self.assertEqual(
            setup.SUPPORTED_HF_MODELS,
            {"microsoft/BitNet-b1.58-2B-4T": "BitNet-b1.58-2B-4T"},
        )

    def test_setup_rejects_unsupported_local_model(self):
        setup = load_module("celiums_setup_local", ROOT / "setup_env.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "config.json"
            config.write_text(json.dumps({"architectures": ["LlamaForCausalLM"], "hidden_act": "silu"}))
            with self.assertRaisesRegex(RuntimeError, "Unsupported strict model architecture"):
                setup.validate_source_model(Path(temp_dir))

    def test_logits_capture_forwards_positions_and_threads(self):
        args = SimpleNamespace(
            model=Path("model.gguf"), prompt="prompt", output=Path("capture.gguf"),
            position=[0, -1], threads=2, threads_batch=8, ctx_size=512,
            batch_size=128, ubatch_size=64, cpu_mask="0x3",
            build_dir=ROOT / "build", tensor=["attn_norm-0", "Qcur-0@ROPE"],
        )
        with patch.object(logits_capture, "capture_binary", return_value=Path("celiums-logits-capture")):
            command = logits_capture.build_command(args)
        self.assertEqual(command[command.index("--threads-batch") + 1], "8")
        self.assertEqual(len([arg for arg in command if arg.startswith("--capture-position=")]), 2)
        self.assertIn("--cpu-strict", command)
        self.assertIn("--cpu-mask-batch", command)
        self.assertIn("--capture-tensor=Qcur-0@ROPE", command)

    def test_release_provenance_is_required_and_version_is_centralized(self):
        manifest = load_module("celiums_release_manifest", ROOT / "utils" / "release_manifest.py")
        self.assertEqual(manifest.VERSION, (ROOT / "VERSION").read_text(encoding="utf-8").strip())
        self.assertEqual(
            manifest.read_required_hash(ROOT / "3rdparty" / "llama.cpp" / "ENGINE_COMMIT"),
            (ROOT / "3rdparty" / "llama.cpp" / "ENGINE_COMMIT").read_text(encoding="utf-8").strip(),
        )
        self.assertEqual(
            manifest.read_required_hash(ROOT / "cmake" / "ENGINE_TREE"),
            (ROOT / "cmake" / "ENGINE_TREE").read_text(encoding="utf-8").strip(),
        )

    def test_release_manifest_uses_relative_artifact_names(self):
        manifest = load_module("celiums_release_paths", ROOT / "utils" / "release_manifest.py")
        self.assertEqual(manifest.display_path(ROOT / "artifact.tar.gz"), "artifact.tar.gz")
        self.assertEqual(manifest.display_path(Path("/tmp/external-artifact.tar.gz")), "external-artifact.tar.gz")


if __name__ == "__main__":
    unittest.main()
