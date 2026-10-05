"""Launch main's Token Yield Studio marketplace next to the Datapoint Factory.

The Studio is ``python -m examples.marketplace_demo_server`` from a clean export
of ``main``. With ``--enable-foundry`` it makes paid calls to the pinned
deployment in ``experiments/customer_requests/pilot.json`` under a campaign
budget. Its campaign state lives in a durable directory outside the export, so
restarts cannot bypass the approved total.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

from source_tree import repository_root, resolve_commit

EXCLUDE = ":(exclude)docs/media"


@dataclass(frozen=True)
class StudioConfig:
    port: int = 8766
    budget_usd: float = 10.0
    approval_id: str = "ginaecho-datapoint-factory-studio-20261005"
    enable_foundry: bool = True

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/marketplace-sales-demo.html"


def export_main(ref: str, cache: Path) -> Path:
    """Full tree of ``ref`` (without large media) in ``cache/studio-<commit>``."""
    repo = repository_root()
    commit = resolve_commit(repo, ref)
    target = cache / f"studio-{commit[:12]}"
    if not (target / ".complete").exists():
        archive = subprocess.check_output(["git", "archive", "--format=tar", commit, "--", ".", EXCLUDE], cwd=repo)
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(target, filter="data")
        (target / ".complete").write_text(commit, encoding="utf-8")
    return target


def ensure_az_account(subscription: str, tenant: str) -> str:
    """Make the pinned subscription the az default, which the Studio's token request uses."""
    az = shutil.which("az") or shutil.which("az.cmd")
    if az is None:
        raise RuntimeError("Azure CLI not found; run az login first")
    current = json.loads(subprocess.check_output([az, "account", "show", "-o", "json"], text=True, timeout=120))
    if current["id"] != subscription or current["tenantId"] != tenant:
        subprocess.check_call([az, "account", "set", "--subscription", subscription], timeout=120)
        current = json.loads(subprocess.check_output([az, "account", "show", "-o", "json"], text=True, timeout=120))
    if current["tenantId"] != tenant:
        raise RuntimeError(f"Subscription {subscription} is not in tenant {tenant}")
    return f"{current['name']} ({current['user']['name']})"


def port_free(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0


def start(root: Path, config: StudioConfig, state_dir: Path, log: Path) -> subprocess.Popen:
    if not port_free(config.port):
        raise RuntimeError(f"Port {config.port} is already in use; choose another with --studio-port")
    command = [sys.executable, "-B", "-m", "examples.marketplace_demo_server", "--port", str(config.port),
               "--run-dir", str(state_dir)]
    if config.enable_foundry:
        command += ["--enable-foundry", "--agent-budget-usd", str(config.budget_usd),
                    "--agent-approval-id", config.approval_id, "--agent-state-dir", str(state_dir / "agent-state")]
    state_dir.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    return subprocess.Popen(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT)
