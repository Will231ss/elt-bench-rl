"""A minimal local implementation of the cookbook's SandboxInterface.

Runs commands in a per-rollout working directory on the host. It is NOT a
security boundary (the agent could `cd ..`); it exists so the environment can
be tested without Docker/Modal. For training, pass a real sandbox through
`sandbox_factory` (e.g. tinker_cookbook.sandbox.ModalSandbox built from an
image with terraform + dbt, or a container on ELT-Bench's docker network).
"""

from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path

from tinker_cookbook.sandbox.sandbox_interface import SandboxResult


class LocalSandbox:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._id = f"local-{uuid.uuid4().hex[:8]}"

    @property
    def sandbox_id(self) -> str:
        return self._id

    def _path(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if self.root not in p.parents and p != self.root:
            raise PermissionError(f"path escapes the workspace: {path}")
        return p

    async def send_heartbeat(self, timeout: int = 30) -> None:
        return None

    async def run_command(
        self, command: str, workdir: str | None = None, timeout: int = 60, max_output_bytes: int | None = None
    ) -> SandboxResult:
        cwd = self._path(workdir) if workdir else self.root
        proc = await asyncio.create_subprocess_shell(
            command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return SandboxResult(stdout="", stderr=f"timed out after {timeout}s", exit_code=124)
        cap = max_output_bytes or 128 * 1024
        return SandboxResult(
            stdout=out.decode(errors="replace")[:cap],
            stderr=err.decode(errors="replace")[:cap],
            exit_code=proc.returncode or 0,
        )

    async def read_file(self, path: str, max_bytes: int | None = None, timeout: int = 60) -> SandboxResult:
        try:
            data = self._path(path).read_text(errors="replace")
        except (OSError, PermissionError) as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)
        return SandboxResult(stdout=data[:max_bytes] if max_bytes else data, stderr="", exit_code=0)

    async def write_file(
        self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60
    ) -> SandboxResult:
        try:
            p = self._path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content if isinstance(content, bytes) else content.encode())
        except (OSError, PermissionError) as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)
        return SandboxResult(stdout="", stderr="", exit_code=0)

    async def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
