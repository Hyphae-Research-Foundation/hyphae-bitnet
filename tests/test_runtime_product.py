import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CMAKE = shutil.which("cmake") or "/tmp/opencode/cmake/bin/cmake"
BUILD_DIR = Path(os.environ.get("CELIUMS_BITNET_TEST_BUILD_DIR", ROOT / "build-runtime"))
INSTALL_PREFIX = os.environ.get("CELIUMS_BITNET_TEST_INSTALL_PREFIX")
EXPECT_GATEWAY = os.environ.get("CELIUMS_BITNET_TEST_EXPECT_GATEWAY") == "1"
MODEL = Path(os.environ.get(
    "CELIUMS_BITNET_TEST_MODEL",
    ROOT / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf",
))
BONSAI_MODEL = Path(os.environ["CELIUMS_BONSAI_TEST_MODEL"]) if os.environ.get(
    "CELIUMS_BONSAI_TEST_MODEL"
) else None
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()


def binary(name="hyphae-bitnet"):
    candidates = [BUILD_DIR / "bin" / name, BUILD_DIR / "bin" / "Release" / f"{name}.exe"]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def clean_environment(**values):
    environment = os.environ.copy()
    environment.pop("CELIUMS_BITNET_API_KEY", None)
    environment.pop("LLAMA_API_KEY", None)
    environment.update(values)
    return environment


class RuntimeProductTests(unittest.TestCase):
    def require_binary(self, name="hyphae-bitnet"):
        path = binary(name)
        if not path.exists():
            self.skipTest(f"product test build is unavailable: {path}")
        return path

    def require_model(self):
        if not MODEL.is_file():
            self.skipTest(f"integration model fixture is unavailable: {MODEL}")

    def require_bonsai_model(self):
        if BONSAI_MODEL is None or not BONSAI_MODEL.is_file():
            self.skipTest("Bonsai integration model fixture is unavailable")

    def test_runtime_cli_identity(self):
        executable = binary()
        if not executable.exists():
            self.skipTest(f"product test build is unavailable: {executable}")
        output = subprocess.run(
            [str(executable), "version"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout
        if f"Hyphae BitNet Runtime {VERSION}" not in output:
            self.skipTest(f"product test build is stale: {executable}")
        self.assertIn(f"Hyphae BitNet Runtime {VERSION}", output)
        self.assertRegex(output, r"product commit: [0-9a-f]{9}")
        self.assertRegex(output, r"engine commit: [0-9a-f]{9}")
        self.assertRegex(output, r"engine tree: [0-9a-f]{40}")
        self.assertIn("strict: true", output)

    def test_runtime_help_lists_public_commands(self):
        output = subprocess.run(
            [str(self.require_binary()), "help"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout
        for command in ("run", "serve", "bench", "validate", "version"):
            self.assertIn(command, output)

    def test_strict_off_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [CMAKE, "-S", str(ROOT), "-B", temp_dir, "-DCELIUMS_BITNET_STRICT=OFF"],
                cwd=ROOT, capture_output=True, text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Non-strict builds are not implemented", result.stdout + result.stderr)

    def test_server_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run([
                CMAKE, "-S", str(ROOT), "-B", temp_dir,
                "-DCELIUMS_BITNET_BUILD_SERVER=OFF", "-DLLAMA_BUILD_TESTS=OFF",
                "-DLLAMA_BUILD_EXAMPLES=OFF", "-DBUILD_SHARED_LIBS=ON",
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            subprocess.run(
                [CMAKE, "--build", temp_dir, "--parallel", "2", "--target", "celiums-bitnet"],
                cwd=ROOT, check=True, capture_output=True, text=True,
            )
            help_output = subprocess.run(
                [str(Path(temp_dir) / "bin" / "hyphae-bitnet"), "help"],
                check=True, capture_output=True, text=True,
            ).stdout
            targets = subprocess.run(
                [CMAKE, "--build", temp_dir, "--target", "help"],
                check=True, capture_output=True, text=True,
            ).stdout
        self.assertNotIn("serve", help_output)
        self.assertNotIn("celiums-runtime-server", targets)

    def test_shared_install_and_cmake_consumer(self):
        if not INSTALL_PREFIX:
            self.skipTest("CELIUMS_BITNET_TEST_INSTALL_PREFIX is not set")
        prefix = Path(INSTALL_PREFIX)
        installed = prefix / "bin" / "hyphae-bitnet"
        self.assertTrue(installed.is_file(), f"installed CLI is unavailable: {installed}")
        self.assertTrue((prefix / "bin" / "celiums-bitnet").exists())
        output = subprocess.run(
            [str(installed), "version"], cwd="/tmp", check=True, capture_output=True, text=True
        ).stdout
        self.assertIn(f"Hyphae BitNet Runtime {VERSION}", output)
        gateway = prefix / "bin" / "celiums-runtime-gateway"
        sidecar = prefix / "bin" / "celiums-hyphae-sidecar"
        mcp = prefix / "bin" / "celiums-runtime-mcp"
        if EXPECT_GATEWAY:
            self.assertTrue(gateway.is_file(), f"installed gateway is unavailable: {gateway}")
            gateway_output = subprocess.run(
                [str(gateway), "version"], cwd="/tmp", check=True,
                capture_output=True, text=True,
            ).stdout
            self.assertIn("Celiums Runtime Gateway", gateway_output)
            self.assertIn("Hyphae 1.2.2", gateway_output)
            self.assertTrue(sidecar.is_file())
            self.assertTrue(mcp.is_file())
        self.assertFalse((prefix / "bin" / "llama-cli").exists())
        self.assertFalse((prefix / "bin" / "llama-server").exists())

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source"
            build = Path(temp_dir) / "build"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.22)\n"
                "project(consumer C)\n"
                "find_package(CeliumsBitNetRuntime CONFIG REQUIRED)\n"
                "add_executable(consumer main.c)\n"
                "target_link_libraries(consumer PRIVATE Celiums::BitNetRuntime)\n",
                encoding="utf-8",
            )
            (source / "main.c").write_text(
                '#include <celiums/bitnet_runtime.h>\n#include <stdio.h>\n'
                'int main(void) { puts(celiums_bitnet_version()); return 0; }\n',
                encoding="utf-8",
            )
            subprocess.run(
                [CMAKE, "-S", str(source), "-B", str(build), f"-DCMAKE_PREFIX_PATH={prefix}"],
                check=True, capture_output=True, text=True,
            )
            subprocess.run([CMAKE, "--build", str(build)], check=True, capture_output=True, text=True)
            consumer = subprocess.run(
                [str(build / "consumer")], check=True, capture_output=True, text=True
            ).stdout.strip()
        self.assertEqual(consumer, VERSION)

        library_dir = next(
            path for path in (
                prefix / "lib" / "celiums-bitnet-runtime",
                prefix / "lib64" / "celiums-bitnet-runtime",
            ) if path.is_dir()
        )
        library = library_dir / "libceliums-bitnet-runtime.so"
        self.assertTrue(library.exists(), f"installed runtime library is unavailable: {library}")
        with tempfile.TemporaryDirectory() as temp_dir:
            old_client = Path(temp_dir) / "old-client"
            subprocess.run([
                os.environ.get("CC", "cc"), "-std=c11",
                f"-I{ROOT / 'tests' / 'abi' / 'v0.3.0' / 'include'}",
                str(ROOT / "tests" / "abi" / "v0.3.0" / "old-client.c"),
                str(library), f"-Wl,-rpath,{library_dir}", "-o", str(old_client),
            ], cwd="/tmp", check=True, capture_output=True, text=True)
            subprocess.run([str(old_client)], cwd=temp_dir, check=True)

        licenses = prefix / "share" / "celiums-bitnet-runtime"
        required = {
            "LICENSE", "LICENSE-MIT", "LICENSE-LLAMA-MIT", "LICENSE-BSD-3-Clause",
            "LICENSE-CPP-HTTPLIB", "LICENSE-jsonhpp", "NOTICE", "NOTICE-CELIUMS",
        }
        if EXPECT_GATEWAY:
            required.update({
                "LICENSE-HYPHAE-APACHE-2.0", "NOTICE-HYPHAE",
                "NOTICE-RUST-DEPENDENCIES", "Cargo.lock", "THIRD_PARTY_LICENSES.html",
            })
        self.assertTrue(required.issubset(path.name for path in licenses.iterdir()))

    def assert_server_policy(self, executable, args, environment=None, expected=2):
        result = subprocess.run(
            [str(executable), *args], cwd=ROOT, env=environment or clean_environment(),
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def start_server(self, extra_args=(), environment=None):
        self.require_model()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        process = subprocess.Popen([
            str(self.require_binary()), "serve", "--model", str(MODEL), "--host", "127.0.0.1",
            "--port", str(port), "-c", "128", "-b", "64", "-ub", "64", "-t", "1", "-tb", "1",
            *extra_args,
        ], cwd=ROOT, env=environment or clean_environment(),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            if process.poll() is not None:
                self.fail("runtime server exited before becoming ready")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except urllib.error.HTTPError as error:
                error.close()
                break
            except OSError:
                time.sleep(0.25)
        else:
            process.terminate()
            self.fail("runtime server did not become ready")
        return process, port

    def stop_server(self, process):
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def test_server_rejects_unauthenticated_remote_binding_from_both_entries(self):
        for executable, prefix in (
            (self.require_binary(), ["serve"]),
            (self.require_binary("celiums-runtime-server"), []),
        ):
            result = self.assert_server_policy(executable, [*prefix, "--host", "0.0.0.0"])
            self.assertIn("refusing unauthenticated remote host", result.stderr)

    def test_server_accepts_both_environment_key_names(self):
        executable = self.require_binary()
        for name in ("CELIUMS_BITNET_API_KEY", "LLAMA_API_KEY"):
            result = self.assert_server_policy(
                executable, ["serve", "--host", "0.0.0.0"], clean_environment(**{name: "secret"})
            )
            self.assertNotIn("refusing unauthenticated remote host", result.stderr)
            self.assertEqual(result.returncode, 2)

    def test_server_rejects_loopback_prefix_hostname_spoof(self):
        result = self.assert_server_policy(
            self.require_binary(), ["serve", "--host", "127.invalid"]
        )
        self.assertIn("refusing unauthenticated remote host", result.stderr)

    def test_server_accepts_api_key_file_and_explicit_override(self):
        executable = self.require_binary()
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as key_file:
            key_file.write("secret\n")
            key_file.flush()
            cases = (
                ["serve", "--host", "0.0.0.0", "--api-key", "secret"],
                ["serve", "--host", "0.0.0.0", "--api-key-file", key_file.name],
                ["serve", "--host", "0.0.0.0", "--allow-unauthenticated-remote"],
            )
            for args in cases:
                result = self.assert_server_policy(executable, args)
                self.assertNotIn("refusing unauthenticated remote host", result.stderr)

    def test_server_enforces_each_api_key_source(self):
        self.require_model()
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as key_file:
            key_file.write("secret\n")
            key_file.flush()
            cases = (
                ((), clean_environment(CELIUMS_BITNET_API_KEY="secret")),
                ((), clean_environment(LLAMA_API_KEY="secret")),
                (("--api-key", "secret"), clean_environment()),
                (("--api-key-file", key_file.name), clean_environment()),
            )
            for extra_args, environment in cases:
                process, port = self.start_server(extra_args, environment)
                try:
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
                    self.assertEqual(error.exception.code, 401)
                    error.exception.close()
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{port}/health",
                        headers={"Authorization": "Bearer secret"},
                    )
                    with urllib.request.urlopen(request, timeout=5) as response:
                        self.assertEqual(response.status, 200)
                finally:
                    self.stop_server(process)

    def test_server_explicit_remote_override_disables_authentication(self):
        process, port = self.start_server(("--allow-unauthenticated-remote",), clean_environment())
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                self.assertEqual(response.status, 200)
        finally:
            self.stop_server(process)

    def test_native_run_generates_expected_greedy_prefix(self):
        self.require_model()
        output = subprocess.run([
            str(self.require_binary()), "run", "--model", str(MODEL), "--prompt", "Hello",
            "-n", "4", "--temp", "0", "-t", "1", "-tb", "1", "-c", "128", "-b", "64", "-ub", "64",
        ], cwd=ROOT, check=True, capture_output=True, text=True).stdout
        self.assertEqual(output, ", I am a\n")

    def test_native_benchmark_reports_prefill_and_decode(self):
        self.require_model()
        output = subprocess.run([
            str(self.require_binary()), "bench", "--model", str(MODEL),
            "-p", "4", "-n", "2", "-t", "1", "-r", "1", "-b", "8", "-ub", "8",
        ], cwd=ROOT, check=True, capture_output=True, text=True).stdout
        rows = [json.loads(line) for line in output.splitlines()]
        self.assertEqual([row["test"] for row in rows], ["pp4", "tg2"])
        self.assertTrue(all(row["model_family"] == "bitnet-b1.58-i2_s" for row in rows))

    def test_bonsai_requires_explicit_family_and_runs(self):
        self.require_bonsai_model()
        rejected = subprocess.run([
            str(self.require_binary()), "validate", "--model", str(BONSAI_MODEL),
        ], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("unsupported strict model", rejected.stderr)

        validated = subprocess.run([
            str(self.require_binary()), "validate", "--model", str(BONSAI_MODEL),
            "--model-family", "bonsai",
        ], cwd=ROOT, check=True, capture_output=True, text=True).stdout
        self.assertIn("model_family: bonsai-qwen35-q1_0", validated)

        rows = subprocess.run([
            str(self.require_binary()), "bench", "--model", str(BONSAI_MODEL),
            "--model-family", "bonsai", "-p", "4", "-n", "1", "-t", "1",
            "-r", "1", "-b", "8", "-ub", "8",
        ], cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
        self.assertEqual([json.loads(row)["model_family"] for row in rows], [
            "bonsai-qwen35-q1_0", "bonsai-qwen35-q1_0",
        ])

    def test_native_server_openai_completion_and_authentication(self):
        self.require_model()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        process = subprocess.Popen([
            str(self.require_binary()), "serve", "--model", str(MODEL), "--host", "127.0.0.1",
            "--port", str(port), "-c", "128", "-b", "64", "-ub", "64", "-t", "1", "-tb", "1",
        ], cwd=ROOT, env=clean_environment(CELIUMS_BITNET_API_KEY="secret"),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(120):
                try:
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{port}/health", headers={"Authorization": "Bearer secret"}
                    )
                    with urllib.request.urlopen(request, timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(0.25)
            else:
                self.fail("runtime server did not become ready")

            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            self.assertEqual(error.exception.code, 401)
            error.exception.close()

            headers = {"Content-Type": "application/json", "Authorization": "Bearer secret"}
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/completions",
                data=json.dumps({"prompt": "Hello", "max_tokens": 2, "temperature": 0}).encode(),
                headers=headers,
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.load(response)
            self.assertEqual(body["choices"][0]["text"], ", I")

            stream_request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/completions",
                data=json.dumps({"prompt": "Hello", "max_tokens": 2, "temperature": 0, "stream": True}).encode(),
                headers=headers,
            )
            with urllib.request.urlopen(stream_request, timeout=120) as response:
                stream_body = response.read().decode()
                self.assertEqual(response.headers.get_content_type(), "text/event-stream")
            events = [line[6:] for line in stream_body.splitlines() if line.startswith("data: ")]
            self.assertEqual(events[-1], "[DONE]")
            self.assertEqual("".join(json.loads(event)["choices"][0]["text"] for event in events[:-1]), ", I")
        finally:
            self.stop_server(process)


if __name__ == "__main__":
    unittest.main()
