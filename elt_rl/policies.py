"""Scripted policies for exercising the environment without a model.

`oracle` replays a correct pipeline; the others are deliberate attempts to get
reward without doing the task. They are what the local integration test runs.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pandas as pd

from tinker_cookbook.renderers.base import ToolCall

from elt_rl.task import ELTTask

Action = tuple[str, dict]


def _load_raw(task: ELTTask) -> list[Action]:
    acts = []
    for spec in task.config.get("flat_files", []):
        acts.append(
            ("sql", {"query": f"CREATE OR REPLACE TABLE raw.{spec['table']} AS "
                              f"SELECT * FROM read_csv_auto('{spec['path']}')"})
        )
    return acts


def _write_gold_sql(task: ELTTask) -> list[Action]:
    return [
        ("write_file", {"path": f"models/{m}.sql", "content": (task.gold_sql_dir / f"{m}.sql").read_text()})
        for m in task.model_names
    ]


def oracle(task: ELTTask, transform_only: bool = False) -> list[Action]:
    acts = [] if transform_only else _load_raw(task)
    return acts + _write_gold_sql(task) + [("run_models", {}), ("submit", {})]


def noop(task: ELTTask, transform_only: bool = False) -> list[Action]:
    return [("submit", {})]


def partial(task: ELTTask, transform_only: bool = False) -> list[Action]:
    """Correct pipeline except one column of the first model is wrong."""
    acts = oracle(task, transform_only)
    first = task.models[0]
    target = f"models/{first.name}.sql"
    out = []
    for name, args in acts:
        if name == "write_file" and args["path"] == target:
            last_col = first.columns[-1][0]
            args = {"path": target, "content": f"SELECT * REPLACE (NULL AS {last_col}) FROM ({args['content']})"}
        out.append((name, args))
    return out


def _literal_select(df: pd.DataFrame) -> str:
    def lit(v):
        if pd.isna(v):
            return "NULL"
        if isinstance(v, str):
            return "'" + v.replace("'", "''") + "'"
        return repr(v.item() if hasattr(v, "item") else v)

    rows = ", ".join("(" + ", ".join(lit(v) for v in row) + ")" for row in df.itertuples(index=False))
    return f"SELECT * FROM (VALUES {rows}) AS t({', '.join(df.columns)})"


def hardcode_tables(task: ELTTask, transform_only: bool = False) -> list[Action]:
    """Loads raw correctly, then types the answer straight into the model tables
    (as if it had looked the answer up). No models/*.sql left behind."""
    acts = [] if transform_only else _load_raw(task)
    for m in task.model_names:
        gold = pd.read_csv(task.gold_csv_dir / f"{m}.csv")
        acts.append(("sql", {"query": f"CREATE OR REPLACE TABLE models.{m} AS {_literal_select(gold)}"}))
    return acts + [("submit", {})]


def hardcode_sql(task: ELTTask, transform_only: bool = False) -> list[Action]:
    """Same, but the literals are written as the 'pipeline' so the rebuild check passes."""
    acts = [] if transform_only else _load_raw(task)
    for m in task.model_names:
        gold = pd.read_csv(task.gold_csv_dir / f"{m}.csv")
        acts.append(("write_file", {"path": f"models/{m}.sql", "content": _literal_select(gold)}))
    return acts + [("run_models", {}), ("submit", {})]


def fake_load(task: ELTTask, transform_only: bool = False) -> list[Action]:
    """Creates raw tables with the right row counts but made-up content."""
    acts = []
    for table, n in task.raw_row_counts.items():
        acts.append(("sql", {"query": f"CREATE OR REPLACE TABLE raw.{table} AS SELECT range AS id FROM range({n})"}))
    return acts + [("submit", {})]


def read_gold(task: ELTTask, transform_only: bool = False) -> list[Action]:
    """Tries to read the grader's gold files through DuckDB."""
    m = task.model_names[0]
    path = (task.gold_csv_dir / f"{m}.csv").resolve().as_posix()
    return oracle(task, transform_only)[:-2] + [
        ("write_file", {"path": f"models/{m}.sql", "content": f"SELECT * FROM read_csv_auto('{path}')"}),
        ("run_models", {}),
        ("submit", {}),
    ]


POLICIES: dict[str, Callable[..., list[Action]]] = {
    "oracle": oracle,
    "noop": noop,
    "partial": partial,
    "hardcode_tables": hardcode_tables,
    "hardcode_sql": hardcode_sql,
    "fake_load": fake_load,
    "read_gold": read_gold,
}


def to_message(i: int, action: Action) -> dict:
    name, args = action
    call = ToolCall(id=f"call_{i}", function=ToolCall.FunctionBody(name=name, arguments=json.dumps(args)))
    return {"role": "assistant", "content": "", "tool_calls": [call]}


async def play(message_env, actions: list[Action]) -> tuple[float, dict, int]:
    """Drive a MessageEnv with a fixed action list. Returns (reward, metrics, steps)."""
    await message_env.initial_observation()
    result = None
    for i, action in enumerate(actions):
        result = await message_env.step(to_message(i, action))
        if result.episode_done:
            return result.reward, result.metrics, i + 1
    return (result.reward if result else 0.0), (result.metrics if result else {}), len(actions)
