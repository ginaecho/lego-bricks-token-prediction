"""Measured non-interactive dispatch through GitHub Copilot CLI."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .runner import AgentResult


def parse_usage(data: Dict[str, Any], model: str) -> Dict[str, int]:
    """Extract provider-reported usage for one fixed model."""

    metrics = data.get("modelMetrics", {}).get(model)
    if not isinstance(metrics, dict):
        raise ValueError(f"usage does not contain model metrics for {model}")
    usage = metrics.get("usage", {})
    required = ("inputTokens", "outputTokens")
    if not all(isinstance(usage.get(key), int) for key in required):
        raise ValueError("usage is missing integer input/output token counts")
    input_tokens = usage["inputTokens"]
    output_tokens = usage["outputTokens"]
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("usage token counts must be non-negative")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


class CopilotDispatcher:
    """Dispatch one fresh CLI session and retain its raw usage artifact."""

    def __init__(
        self,
        artifact_dir: Path,
        label: str,
        model: str = "claude-haiku-4.5",
        executable: Optional[str] = None,
        timeout_seconds: int = 300,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.label = label
        self.model = model
        if executable:
            self.command_prefix = [executable]
        elif os.name == "nt":
            loader = (
                Path(os.environ["APPDATA"])
                / "npm"
                / "node_modules"
                / "@github"
                / "copilot"
                / "npm-loader.js"
            )
            if not loader.is_file():
                raise FileNotFoundError(f"Copilot Node launcher not found: {loader}")
            self.command_prefix = ["node", str(loader)]
        else:
            self.command_prefix = ["copilot"]
        self.timeout_seconds = timeout_seconds

    def __call__(self, prompt: str) -> AgentResult:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        usage_path = self.artifact_dir / f"{self.label}_usage.json"
        output_path = self.artifact_dir / f"{self.label}_output.txt"
        command = [
            *self.command_prefix,
            "-p",
            prompt,
            "--model",
            self.model,
            "--no-custom-instructions",
            "--disable-builtin-mcps",
            "--no-remote-export",
            "--allow-all-tools",
            "--silent",
            "--usage-output-file",
            str(usage_path),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
            check=False,
        )
        output_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(
                f"Copilot dispatch failed ({completed.returncode}): "
                f"{completed.stderr.strip()}"
            )
        if not usage_path.is_file():
            raise RuntimeError("Copilot dispatch did not write usage statistics")
        usage_data = json.loads(usage_path.read_text(encoding="utf-8"))
        usage = parse_usage(usage_data, self.model)
        model_usage = usage_data["modelMetrics"][self.model]["usage"]
        return AgentResult(
            output=completed.stdout.strip(),
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            tool_uses=None,
            duration_ms=int(usage_data.get("totalApiDurationMs", 0)),
            model=self.model,
            model_version=self.model,
        )
