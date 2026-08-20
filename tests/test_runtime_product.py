import json
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CMAKE = Path("/tmp/opencode/cmake/bin/cmake")


class RuntimeProductTests(unittest.TestCase):
    def test_runtime_cli_identity(self):
        binary = ROOT / "build-runtime" / "bin" / "celiums-bitnet"
        if not binary.exists():
            self.skipTest("celiums-bitnet was not built")
        output = subprocess.run(
            [str(binary), "version"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout
        self.assertIn("Celiums BitNet Runtime 0.2.0", output)
        self.assertRegex(output, r"product commit: [0-9a-f]{9}")
        self.assertRegex(output, r"engine commit: [0-9a-f]{9}")
        self.assertIn("strict: true", output)

    def test_runtime_help_lists_public_commands(self):
        binary = ROOT / "build-runtime" / "bin" / "celiums-bitnet"
        if not binary.exists():
            self.skipTest("celiums-bitnet was not built")
        output = subprocess.run(
            [str(binary), "help"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout
        for command in ("run", "serve", "bench", "validate", "version"):
            self.assertIn(command, output)

    def test_strict_off_is_rejected(self):
        if not CMAKE.exists():
            self.skipTest("cmake was not found")
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [str(CMAKE), "-S", str(ROOT), "-B", temp_dir, "-DCELIUMS_BITNET_STRICT=OFF"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Non-strict builds are not implemented", result.stdout + result.stderr)

    def test_tl_requires_experimental(self):
        if not CMAKE.exists():
            self.skipTest("cmake was not found")
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [str(CMAKE), "-S", str(ROOT), "-B", temp_dir, "-DBITNET_X86_TL2=ON"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("require CELIUMS_BITNET_EXPERIMENTAL=ON", result.stdout + result.stderr)

    def test_server_can_be_disabled(self):
        if not CMAKE.exists():
            self.skipTest("cmake was not found")
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    str(CMAKE), "-S", str(ROOT), "-B", temp_dir,
                    "-DCELIUMS_BITNET_BUILD_SERVER=OFF",
                    "-DLLAMA_BUILD_COMMON=ON",
                    "-DLLAMA_BUILD_TOOLS=ON",
                    "-DLLAMA_BUILD_TESTS=OFF",
                    "-DLLAMA_BUILD_EXAMPLES=OFF",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_default_install_is_relocatable_and_private(self):
        binary = Path("/tmp/opencode/celiums-runtime-install/bin/celiums-bitnet")
        if not binary.exists():
            self.skipTest("runtime install smoke prefix was not created")
        output = subprocess.run(
            [str(binary), "version"], cwd="/tmp", check=True, capture_output=True, text=True
        ).stdout
        self.assertIn("Celiums BitNet Runtime 0.2.0", output)
        self.assertRegex(output, r"engine commit: [0-9a-f]{9}")
        self.assertFalse((binary.parent / "llama-cli").exists())
        self.assertFalse((binary.parent / "llama-server").exists())
        self.assertFalse((binary.parent / "celiums-logits-capture").exists())

    def test_installed_cmake_package_builds_c_consumer(self):
        prefix = Path("/tmp/opencode/celiums-runtime-install")
        source = Path("/tmp/opencode/celiums-runtime-consumer")
        if not prefix.exists() or not source.exists() or not CMAKE.exists():
            self.skipTest("runtime CMake consumer fixture is unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [str(CMAKE), "-S", str(source), "-B", temp_dir, f"-DCMAKE_PREFIX_PATH={prefix}"],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [str(CMAKE), "--build", temp_dir, "--parallel", "2"],
                check=True,
                capture_output=True,
                text=True,
            )
            output = subprocess.run(
                [str(Path(temp_dir) / "celiums-runtime-consumer")],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        self.assertRegex(output.strip(), r"^0\.2\.0 [0-9a-f]{9}$")

    def test_install_contains_required_notices(self):
        licenses = Path("/tmp/opencode/celiums-runtime-install/share/celiums-bitnet-runtime")
        if not licenses.exists():
            self.skipTest("runtime install smoke prefix was not created")
        required = {
            "LICENSE",
            "LICENSE-MIT",
            "LICENSE-LLAMA-MIT",
            "LICENSE-BSD-3-Clause",
            "LICENSE-CPP-HTTPLIB",
            "LICENSE-jsonhpp",
            "NOTICE",
            "NOTICE-CELIUMS",
        }
        self.assertTrue(required.issubset(path.name for path in licenses.iterdir()))

    def test_server_rejects_unauthenticated_remote_binding(self):
        binary = ROOT / "build-runtime" / "bin" / "celiums-bitnet"
        if not binary.exists():
            self.skipTest("celiums-bitnet was not built")
        result = subprocess.run(
            [str(binary), "serve", "--host", "0.0.0.0"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("refusing unauthenticated remote host", result.stderr)

    def test_native_run_generates_expected_greedy_prefix(self):
        binary = ROOT / "build-runtime" / "bin" / "celiums-bitnet"
        model = ROOT / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf"
        if not binary.exists() or not model.exists():
            self.skipTest("runtime binary or model is unavailable")
        output = subprocess.run(
            [
                str(binary), "run", "--model", str(model), "--prompt", "Hello",
                "-n", "4", "--temp", "0", "-t", "1", "-tb", "1",
                "-c", "128", "-b", "64", "-ub", "64",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertEqual(output, ", I am a\n")

    def test_native_benchmark_reports_prefill_and_decode(self):
        binary = ROOT / "build-runtime" / "bin" / "celiums-bitnet"
        model = ROOT / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf"
        if not binary.exists() or not model.exists():
            self.skipTest("runtime binary or model is unavailable")
        output = subprocess.run(
            [
                str(binary), "bench", "--model", str(model),
                "-p", "4", "-n", "2", "-t", "1", "-r", "1", "-b", "8", "-ub", "8",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        rows = [json.loads(line) for line in output.splitlines()]
        self.assertEqual([row["test"] for row in rows], ["pp4", "tg2"])

    def test_native_server_openai_completion(self):
        binary = ROOT / "build-runtime" / "bin" / "celiums-bitnet"
        model = ROOT / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf"
        if not binary.exists() or not model.exists():
            self.skipTest("runtime binary or model is unavailable")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        process = subprocess.Popen(
            [
                str(binary), "serve", "--model", str(model), "--host", "127.0.0.1",
                "--port", str(port), "-c", "128", "-b", "64", "-ub", "64", "-t", "1", "-tb", "1",
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            for _ in range(120):
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(0.25)
            else:
                self.fail("runtime server did not become ready")
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/completions",
                data=json.dumps({"prompt": "Hello", "max_tokens": 2, "temperature": 0}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.load(response)
            self.assertEqual(body["object"], "text_completion")
            self.assertEqual(body["choices"][0]["text"], ", I")

            stream_request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/completions",
                data=json.dumps(
                    {"prompt": "Hello", "max_tokens": 2, "temperature": 0, "stream": True}
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(stream_request, timeout=120) as response:
                stream_body = response.read().decode()
                self.assertEqual(response.headers.get_content_type(), "text/event-stream")
            events = [line[6:] for line in stream_body.splitlines() if line.startswith("data: ")]
            self.assertEqual(events[-1], "[DONE]")
            chunks = [json.loads(event) for event in events[:-1]]
            text = "".join(choice["choices"][0]["text"] for choice in chunks)
            self.assertEqual(text, ", I")

            chat_request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(
                    {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 0}
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(chat_request, timeout=120) as response:
                chat_body = json.load(response)
            self.assertEqual(chat_body["object"], "chat.completion")

            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as response:
                metrics = response.read().decode()
            self.assertIn("celiums_bitnet_http_requests_total 3", metrics)
            self.assertIn("celiums_bitnet_completions_total 3", metrics)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    unittest.main()
