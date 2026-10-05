"""Workspace for the live builder: four allowed files, bounded test and CLI runs.

Model-written code runs locally in a dedicated directory with timeouts, an
isolated interpreter (-I) and a minimal environment without credentials.
This limits accidents; it is not a security sandbox.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

FILES = ("implementation.py", "test_implementation.py", "example_input.json", "build_manifest.json")
MAX_BYTES = 200_000
TEST_COMMAND = [sys.executable, "-I", "-B", "-m", "unittest", "discover", "-s", ".", "-p", "test_implementation.py"]
CLI_COMMAND = [sys.executable, "-I", "-B", "implementation.py", "example_input.json"]
KEEP_ENV = ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "PATHEXT", "COMSPEC")


class Workspace:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, content: str) -> str:
        if name not in FILES:
            return f"error: only {', '.join(FILES)} may be written"
        if len(content.encode("utf-8")) > MAX_BYTES:
            return f"error: file exceeds {MAX_BYTES} bytes"
        (self.directory / name).write_text(content, encoding="utf-8")
        return f"wrote {name} ({content.count(chr(10)) + 1} lines)"

    def read(self, name: str) -> str:
        path = self.directory / name
        return path.read_text(encoding="utf-8") if name in FILES and path.is_file() else f"error: {name} not found"

    def _run(self, command: list[str], timeout: int) -> dict:
        env = {key: os.environ[key] for key in KEEP_ENV if key in os.environ} | {"PYTHONIOENCODING": "utf-8"}
        try:
            done = subprocess.run(command, cwd=self.directory, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=timeout, env=env)
            return {"exit_code": done.returncode, "stdout": done.stdout, "stderr": done.stderr}
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "stdout": "", "stderr": f"timed out after {timeout} s"}

    def run_tests(self) -> dict:
        return self._run(TEST_COMMAND, 120)

    def run_cli(self) -> dict:
        return self._run(CLI_COMMAND, 30)

    def files(self) -> dict:
        return {path.name: {"bytes": path.stat().st_size, "lines": path.read_text(encoding="utf-8").count("\n")}
                for path in sorted(self.directory.iterdir()) if path.name in FILES}

    def verify(self) -> dict:
        """Independent re-run by the app, never the builder's own claim."""
        missing = [name for name in FILES if not (self.directory / name).is_file()]
        tests, cli = self.run_tests(), self.run_cli()
        try:
            cli_output, cli_json = json.loads(cli["stdout"]), True
        except json.JSONDecodeError:
            cli_output, cli_json = cli["stdout"][-2000:], False
        passed = not missing and tests["exit_code"] == 0 and cli["exit_code"] == 0 and cli_json
        return {
            "tests_passed": tests["exit_code"] == 0, "accepted": passed, "missing_files": missing,
            "test_command": "python -B -m unittest discover -s . -p test_implementation.py",
            "test_output": (tests["stdout"] + tests["stderr"])[-4000:],
            "cli_exit_code": cli["exit_code"], "cli_arguments": ["example_input.json"], "cli_output": cli_output,
            "artifact_sha256": {name: hashlib.sha256((self.directory / name).read_bytes()).hexdigest()
                                for name in FILES if (self.directory / name).is_file()},
        }
