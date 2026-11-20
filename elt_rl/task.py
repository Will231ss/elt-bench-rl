"""Loading ELT-Bench tasks (and the local fixture tasks, which use the same layout)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Files the agent is never allowed to see. Everything under a task's `_gold/`
# directory stays on the grader side.
GOLD_DIRNAME = "_gold"


@dataclass(frozen=True)
class DataModel:
    name: str
    description: str
    columns: list[tuple[str, str]]  # (name, description)


@dataclass(frozen=True)
class ELTTask:
    """One ELT pipeline to build.

    `project_dir` is the project base the agent starts from (config.yaml,
    data_model.yaml, schemas/, documentation/, elt/main.tf for real tasks).
    The grader-only pieces (expected row counts, sort keys, gold outputs) are
    loaded separately and never copied into the agent's workspace.
    """

    name: str
    project_dir: Path
    config: dict
    models: list[DataModel]
    raw_row_counts: dict[str, int]  # expected raw table -> row count (table.json)
    sort_keys: dict[str, list[str]]  # model -> unique key columns (sort_key.json)
    gold_csv_dir: Path | None = None  # <model>.csv with the expected output
    gold_sql_dir: Path | None = None  # <model>.sql, only for local fixtures
    extra: dict = field(default_factory=dict)

    @property
    def model_names(self) -> list[str]:
        return [m.name for m in self.models]

    def agent_files(self) -> list[Path]:
        """Files copied into the agent workspace (relative to project_dir)."""
        out = []
        for p in sorted(self.project_dir.rglob("*")):
            rel = p.relative_to(self.project_dir)
            if p.is_file() and rel.parts[0] != GOLD_DIRNAME:
                out.append(rel)
        return out


def _load_models(path: Path) -> list[DataModel]:
    spec = yaml.safe_load(path.read_text())
    models = []
    for m in spec["models"]:
        cols = [(c["name"].strip(), c.get("description", "").strip()) for c in m["columns"]]
        models.append(DataModel(m["name"], m.get("description", ""), cols))
    return models


def load_fixture_task(task_dir: str | Path) -> ELTTask:
    """Local task in ELT-Bench layout plus a `_gold/` folder (see fixtures/)."""
    task_dir = Path(task_dir)
    gold = task_dir / GOLD_DIRNAME
    return ELTTask(
        name=task_dir.name,
        project_dir=task_dir,
        config=yaml.safe_load((task_dir / "config.yaml").read_text()),
        models=_load_models(task_dir / "data_model.yaml"),
        raw_row_counts=json.loads((gold / "table.json").read_text()),
        sort_keys=json.loads((gold / "sort_key.json").read_text()),
        gold_csv_dir=gold / "expected",
        gold_sql_dir=gold / "models",
    )


def load_eltbench_task(
    eltbench_root: str | Path,
    name: str,
    destination: str = "snowflake",
    gt_dir: str | Path | None = None,
) -> ELTTask:
    """Official task from an ELT-Bench checkout (branch elt_bench++).

    Task specs live in elt-bench/<destination>/<name>/, the expected row counts
    and sort keys in evaluation/table.json and evaluation/sort_key.json. The
    ground-truth CSVs come from the gt zip that setup/elt_setup.sh downloads
    (evaluation/gt_<destination>/<name>/<model>.csv by default).
    """
    root = Path(eltbench_root)
    task_dir = root / "elt-bench" / destination / name
    if not task_dir.exists():
        raise FileNotFoundError(task_dir)
    table_json = json.loads((root / "evaluation" / "table.json").read_text())
    sort_json = json.loads((root / "evaluation" / "sort_key.json").read_text())
    gt_root = Path(gt_dir) if gt_dir else root / "evaluation" / f"gt_{destination}"
    return ELTTask(
        name=name,
        project_dir=task_dir,
        config=yaml.safe_load((task_dir / "config.yaml").read_text()),
        models=_load_models(task_dir / "data_model.yaml"),
        raw_row_counts=table_json[name],
        sort_keys=sort_json.get(name, {}),
        gold_csv_dir=gt_root / name,
        extra={"eltbench_root": str(root), "destination": destination},
    )
