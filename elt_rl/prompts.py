from __future__ import annotations

from elt_rl.task import ELTTask

SYSTEM_PROMPT = """You are a data engineer building an ELT pipeline.
You work inside a project directory and have tools to run shell commands, read and write files, \
and run SQL against your own warehouse namespace. Tool outputs are truncated, so query small \
samples instead of whole tables.

The pipeline has two stages:
1. Extract & load: get every source table listed in config.yaml into the raw schema, unchanged.
2. Transform: build every data model described in data_model.yaml in the model schema. \
Each model must have exactly the columns listed, one row per entity.

Your pipeline is graded by executing it, not by what you say. When everything is built and \
checked, call `submit`."""


def task_brief(task: ELTTask, warehouse_note: str, mode: str, stage1_hint: str) -> str:
    lines = [f"Task: `{task.name}`.", warehouse_note, ""]
    if mode == "transform_only":
        lines.append(
            "The raw source tables are already loaded into the raw schema. "
            "Only stage 2 (transform) is needed."
        )
    else:
        lines.append(stage1_hint)
    lines += [
        "",
        "Project files you should read first: config.yaml (sources and connections), "
        "data_model.yaml (target models and column definitions), schemas/ (source columns)"
        + (", documentation/ and elt/main.tf." if (task.project_dir / "elt").exists() else "."),
        "",
        "Data models to build: " + ", ".join(f"`{m}`" for m in task.model_names) + ".",
    ]
    return "\n".join(lines)


DUCKDB_STAGE1_HINT = (
    "Stage 1: the sources are flat files under sources/ (see config.yaml). Load each one into the "
    "raw schema with the same table name, e.g. "
    "`CREATE TABLE raw.book AS SELECT * FROM read_csv_auto('sources/book.csv')`. "
    "Stage 2: write each model as a single SELECT in models/<model_name>.sql, then call `run_models` "
    "to build them and check the results with `sql`."
)

AIRBYTE_STAGE1_HINT = (
    "Stage 1: write Terraform in elt/ that configures Airbyte sources, the Snowflake destination "
    "and one connection per source (follow documentation/), run `terraform init` and "
    "`terraform apply`, then trigger the syncs through the Airbyte API and wait for them to finish. "
    "Stage 2: set up a dbt project for Snowflake, write one model per data model, run `dbt run`, "
    "and check the output tables with `sql`."
)
