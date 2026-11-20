"""Tools exposed to the agent during an ELT episode."""

from __future__ import annotations

import json
from typing import Annotated

from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool

from elt_rl.warehouse.base import Namespace, Warehouse

MAX_OUTPUT_CHARS = 6000  # long tool outputs blow up the context very quickly
MAX_SQL_ROWS = 30


def _clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [{len(text) - limit} chars cut] ...\n" + text[-half:]


class ELTTools:
    def __init__(
        self,
        sandbox: SandboxInterface,
        warehouse: Warehouse,
        ns: Namespace,
        model_names: list[str],
        command_timeout: int = 300,
    ):
        self.sandbox = sandbox
        self.warehouse = warehouse
        self.ns = ns
        self.model_names = model_names
        self.command_timeout = command_timeout
        self.submitted = False

    @tool
    async def bash(self, command: Annotated[str, "Shell command to run in the project directory."]) -> ToolResult:
        """Run a shell command (terraform, dbt, python, ls, cat, ...) in the project directory."""
        r = await self.sandbox.run_command(command, timeout=self.command_timeout)
        return simple_tool_result(
            _clip(json.dumps({"exit_code": r.exit_code, "stdout": r.stdout, "stderr": r.stderr}))
        )

    @tool
    async def read_file(self, path: Annotated[str, "Path relative to the project directory."]) -> ToolResult:
        """Read a text file from the project directory."""
        r = await self.sandbox.read_file(path)
        return simple_tool_result(_clip(r.stdout if r.exit_code == 0 else f"error: {r.stderr}"))

    @tool
    async def write_file(
        self,
        path: Annotated[str, "Path relative to the project directory."],
        content: Annotated[str, "Full file content."],
    ) -> ToolResult:
        """Create or overwrite a file in the project directory."""
        r = await self.sandbox.write_file(path, content)
        return simple_tool_result("ok" if r.exit_code == 0 else f"error: {r.stderr}")

    @tool
    async def sql(self, query: Annotated[str, "One SQL statement."]) -> ToolResult:
        """Run one SQL statement against your warehouse namespace and see up to 30 result rows."""
        try:
            df = self.warehouse.query(self.ns, query)
        except Exception as e:
            return simple_tool_result(f"error: {type(e).__name__}: {e}")
        if df.empty:
            return simple_tool_result("ok (no rows)")
        text = df.head(MAX_SQL_ROWS).to_csv(index=False)
        if len(df) > MAX_SQL_ROWS:
            text += f"... {len(df) - MAX_SQL_ROWS} more rows"
        return simple_tool_result(_clip(text))

    @tool
    async def run_models(self) -> ToolResult:
        """Build every data model from models/<name>.sql (a SELECT statement each) into the model schema.

        A small stand-in for `dbt run` used with the DuckDB destination.
        """
        lines = []
        pending = list(self.model_names)
        for _ in range(len(pending) + 1):  # models can depend on each other; retry failures
            failed = []
            for model in pending:
                r = await self.sandbox.read_file(f"models/{model}.sql")
                if r.exit_code != 0:
                    lines.append(f"{model}: models/{model}.sql not found")
                    continue
                try:
                    self.warehouse.materialize(self.ns, model, r.stdout.strip().rstrip(";"))
                    lines.append(f"{model}: OK")
                except Exception as e:
                    failed.append(model)
                    last_err = f"{model}: ERROR {type(e).__name__}: {e}"
            if not failed or len(failed) == len(pending):
                if failed:
                    lines.append(last_err)
                break
            pending = failed
        return simple_tool_result(_clip("\n".join(lines)))

    @tool
    async def submit(self) -> ToolResult:
        """Finish the episode. Call this once the raw tables are loaded and all data models are built."""
        self.submitted = True
        return simple_tool_result("Submitted. The pipeline will now be graded.", should_stop=True)

    def all(self, include_run_models: bool = True) -> list:
        tools = [self.bash, self.read_file, self.write_file, self.sql]
        if include_run_models:
            tools.append(self.run_models)
        tools.append(self.submit)
        return tools
