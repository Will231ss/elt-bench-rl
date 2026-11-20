"""ELT-Bench as a tinker-cookbook RL environment.

Structure follows tinker_cookbook/recipes/harbor_rl: one EnvGroupBuilder per
task, `group_size` rollouts per group, each rollout an AgentToolMessageEnv with
its own sandbox, warehouse namespace, tools and grader. Rewards are only given
at the end of the episode, by ELTGrader.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import chz
import yaml

from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.tool_use.agent_tool_message_env import AgentToolMessageEnv

from elt_rl.grader import ELTGrader, RewardWeights
from elt_rl.prompts import AIRBYTE_STAGE1_HINT, DUCKDB_STAGE1_HINT, SYSTEM_PROMPT, task_brief
from elt_rl.sandbox import LocalSandbox
from elt_rl.task import ELTTask
from elt_rl.tools import ELTTools
from elt_rl.warehouse import Warehouse
from elt_rl.warehouse.base import Namespace

logger = logging.getLogger(__name__)

SandboxFactory = Callable[[Path], Awaitable[SandboxInterface]]


async def local_sandbox_factory(workdir: Path) -> SandboxInterface:
    return LocalSandbox(workdir)


@dataclass
class Rollout:
    """Everything one rollout owns. Created fresh per episode, torn down in cleanup()."""

    rollout_id: str
    workspace: Path
    ns: Namespace
    sandbox: SandboxInterface
    tools: ELTTools
    grader: ELTGrader
    initial_messages: list = field(default_factory=list)


async def make_rollout(
    task: ELTTask,
    warehouse: Warehouse,
    workspace_root: Path,
    mode: str = "full",
    sandbox_factory: SandboxFactory = local_sandbox_factory,
    weights: RewardWeights | None = None,
    command_timeout: int = 300,
) -> Rollout:
    rid = uuid.uuid4().hex[:10]
    workspace = Path(workspace_root) / f"{task.name}_{rid}"
    workspace.mkdir(parents=True)
    for rel in task.agent_files():  # never copies _gold/
        dst = workspace / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(task.project_dir / rel, dst)
    (workspace / "models").mkdir(exist_ok=True)

    database = task.config.get("snowflake", {}).get("config", {}).get("database")
    ns = warehouse.create_namespace(rid, database)
    if mode == "transform_only":
        warehouse.preload_raw(ns, task)
    if warehouse.name == "duckdb":
        ns.handle["workdir"] = str(workspace)
    else:
        _point_config_at_namespace(workspace / "config.yaml", ns)

    sandbox = await sandbox_factory(workspace)
    tools = ELTTools(sandbox, warehouse, ns, task.model_names, command_timeout=command_timeout)
    grader = ELTGrader(task, warehouse, ns, workspace, mode=mode, weights=weights or RewardWeights())
    hint = DUCKDB_STAGE1_HINT if warehouse.name == "duckdb" else AIRBYTE_STAGE1_HINT
    messages = [
        {"role": "user", "content": task_brief(task, warehouse.describe_for_agent(ns), mode, hint)}
    ]
    return Rollout(rid, workspace, ns, sandbox, tools, grader, messages)


def _point_config_at_namespace(config_path: Path, ns: Namespace) -> None:
    """Rewrite the workspace copy of config.yaml so Airbyte writes into this
    rollout's raw schema and dbt builds into its model schema."""
    cfg = yaml.safe_load(config_path.read_text())
    sf = cfg.setdefault("snowflake", {}).setdefault("config", {})
    sf["schema"] = ns.raw_schema.split(".")[-1]
    sf["dbt_target_schema"] = ns.model_schema.split(".")[-1]
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def make_message_env(rollout: Rollout, max_turns: int, include_run_models: bool) -> AgentToolMessageEnv:
    """Message-level env. Useful on its own for scripted policies and tests."""
    return AgentToolMessageEnv(
        tools=rollout.tools.all(include_run_models),
        initial_messages=rollout.initial_messages,
        max_turns=max_turns,
        reward_fn=rollout.grader,
    )


async def teardown(rollout: Rollout, warehouse: Warehouse) -> None:
    try:
        warehouse.drop_namespace(rollout.ns)
    except Exception as e:
        logger.warning("dropping namespace %s failed: %s", rollout.ns.rollout_id, e)
    await rollout.sandbox.cleanup()
    shutil.rmtree(rollout.workspace, ignore_errors=True)


class ELTEnvGroupBuilder(EnvGroupBuilder):
    def __init__(
        self,
        task: ELTTask,
        warehouse: Warehouse,
        model_name: str,
        renderer_name: str | None,
        group_size: int,
        workspace_root: Path,
        mode: str = "full",
        max_turns: int = 40,
        max_trajectory_tokens: int = 32 * 1024,
        max_generation_tokens: int | None = None,
        sandbox_factory: SandboxFactory = local_sandbox_factory,
        weights: RewardWeights | None = None,
    ):
        self.task = task
        self.warehouse = warehouse
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.group_size = group_size
        self.workspace_root = Path(workspace_root)
        self.mode = mode
        self.max_turns = max_turns
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self.sandbox_factory = sandbox_factory
        self.weights = weights
        self._rollouts: list[Rollout] = []

    async def make_envs(self) -> Sequence[Env]:
        # Imported here so the rest of the package works without a tokenizer.
        from tinker_cookbook import model_info, tokenizer_utils
        from tinker_cookbook.renderers import get_renderer
        from tinker_cookbook.rl.message_env import EnvFromMessageEnv

        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer = get_renderer(
            self.renderer_name or model_info.get_recommended_renderer_name(self.model_name), tokenizer
        )
        include_run_models = self.warehouse.name == "duckdb"

        envs = []
        for _ in range(self.group_size):
            r = await make_rollout(
                self.task, self.warehouse, self.workspace_root, self.mode, self.sandbox_factory, self.weights
            )
            tool_specs = [t.to_spec() for t in r.tools.all(include_run_models)]
            r.initial_messages = (
                renderer.create_conversation_prefix_with_tools(tools=tool_specs, system_prompt=SYSTEM_PROMPT)
                + r.initial_messages
            )
            self._rollouts.append(r)
            envs.append(
                EnvFromMessageEnv(
                    renderer=renderer,
                    message_env=make_message_env(r, self.max_turns, include_run_models),
                    failed_parse_reward=-0.1,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                    max_generation_tokens=self.max_generation_tokens,
                    context_overflow_reward=-0.1,
                )
            )
        return envs

    async def cleanup(self) -> None:
        for r in self._rollouts:
            await teardown(r, self.warehouse)
        self._rollouts.clear()

    def logging_tags(self) -> list[str]:
        return ["elt", self.mode, self.warehouse.name]


class ELTDataset(RLDataset):
    def __init__(self, builders: list[ELTEnvGroupBuilder], batch_size: int):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = (index * self.batch_size) % max(len(self.builders), 1)
        batch = self.builders[start : start + self.batch_size]
        if len(batch) < self.batch_size:  # wrap around, tasks are few
            batch = batch + self.builders[: self.batch_size - len(batch)]
        return batch

    def __len__(self) -> int:
        return max(1, (len(self.builders) + self.batch_size - 1) // self.batch_size)


@chz.chz
class ELTDatasetBuilder(RLDatasetBuilder):
    tasks: list[ELTTask]
    warehouse: Warehouse
    model_name: str
    batch_size: int
    group_size: int
    workspace_root: str
    renderer_name: str | None = None
    mode: str = "full"
    max_turns: int = 40
    max_trajectory_tokens: int = 32 * 1024
    max_generation_tokens: int | None = None
    sandbox_factory: SandboxFactory = local_sandbox_factory

    def _builders(self, group_size: int) -> list[ELTEnvGroupBuilder]:
        return [
            ELTEnvGroupBuilder(
                task=t,
                warehouse=self.warehouse,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                group_size=group_size,
                workspace_root=Path(self.workspace_root),
                mode=self.mode,
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                sandbox_factory=self.sandbox_factory,
            )
            for t in self.tasks
        ]

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        return ELTDataset(self._builders(self.group_size), self.batch_size), None
