"""RL training entry point (same shape as tinker_cookbook.recipes.harbor_rl.train).

Local smoke run on the fixture (no warehouse credentials, needs TINKER_API_KEY):
    python -m elt_rl.train destination=duckdb tasks=fixtures/tiny_books max_steps=1

One step on an official ELT-Bench task, transform-only, on Snowflake:
    python -m elt_rl.train destination=snowflake eltbench_root=../ELT-Bench \
        tasks=books mode=transform_only credential=./snowflake_credential.json max_steps=1
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import chz

from tinker_cookbook import cli_utils
from tinker_cookbook.rl.train import Config, main

from elt_rl.env import ELTDatasetBuilder
from elt_rl.task import load_eltbench_task, load_fixture_task
from elt_rl.warehouse import get_warehouse

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    # tasks: comma-separated fixture dirs (duckdb) or ELT-Bench task names (snowflake)
    tasks: str = "fixtures/tiny_books"
    destination: str = "duckdb"
    mode: str = "transform_only"
    eltbench_root: str | None = None
    gt_dir: str | None = None
    credential: str | None = None
    snapshot_schema: str = "AIRBYTE_SCHEMA"
    workspace_root: str = "/tmp/elt_rl_runs"

    model_name: str = "Qwen/Qwen3.5-4B"
    renderer_name: str | None = None
    lora_rank: int = 32
    max_tokens: int = 2048
    temperature: float = 1.0

    max_turns: int = 30
    max_trajectory_tokens: int = 32 * 1024
    group_size: int = 4
    groups_per_batch: int = 2
    learning_rate: float = 4e-5
    max_steps: int | None = None

    log_path: str | None = None
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def build(cfg: CLIConfig) -> Config:
    if cfg.destination == "duckdb":
        tasks = [load_fixture_task(p) for p in cfg.tasks.split(",")]
        warehouse = get_warehouse("duckdb", root=Path(cfg.workspace_root) / "duckdb")
    else:
        assert cfg.eltbench_root and cfg.credential, "snowflake needs eltbench_root and credential"
        tasks = [
            load_eltbench_task(cfg.eltbench_root, name, cfg.destination, cfg.gt_dir)
            for name in cfg.tasks.split(",")
        ]
        warehouse = get_warehouse(
            cfg.destination, credential_path=cfg.credential, snapshot_schema=cfg.snapshot_schema
        )

    run_name = f"elt-{cfg.destination}-{cfg.mode}-{cfg.model_name.split('/')[-1]}-{datetime.now():%m%d-%H%M}"
    dataset = ELTDatasetBuilder(
        tasks=tasks,
        warehouse=warehouse,
        model_name=cfg.model_name,
        renderer_name=cfg.renderer_name,
        batch_size=cfg.groups_per_batch,
        group_size=cfg.group_size,
        workspace_root=str(Path(cfg.workspace_root) / "workspaces"),
        mode=cfg.mode,
        max_turns=cfg.max_turns,
        max_trajectory_tokens=cfg.max_trajectory_tokens,
        max_generation_tokens=cfg.max_tokens,
    )
    return Config(
        learning_rate=cfg.learning_rate,
        dataset_builder=dataset,
        model_name=cfg.model_name,
        recipe_name="elt_bench_rl",
        lora_rank=cfg.lora_rank,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        log_path=cfg.log_path or cli_utils.runs_path("elt_rl", run_name),
        base_url=cfg.base_url,
        eval_every=0,
        save_every=0,
        max_steps=cfg.max_steps,
    )


def cli_main(cfg: CLIConfig) -> None:
    config = build(cfg)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    asyncio.run(main(config))


if __name__ == "__main__":
    cli_main(chz.entrypoint(CLIConfig))
