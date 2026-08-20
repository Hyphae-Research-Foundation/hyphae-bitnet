import subprocess
import tempfile
import unittest
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
        self.assertIn("engine commit: 3015cb476", output)
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
        self.assertIn("engine commit: 3015cb476", output)
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
        self.assertEqual(output.strip(), "0.2.0 3015cb476")

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


if __name__ == "__main__":
    unittest.main()
