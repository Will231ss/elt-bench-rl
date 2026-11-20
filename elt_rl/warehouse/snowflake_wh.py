"""Snowflake destination (the one ELT-Bench uses).

Each rollout gets two schemas in the task's database, e.g. BOOKS.R_ab12_RAW
and BOOKS.R_ab12_MODELS. The agent's Airbyte destination and dbt target are
pointed at those schemas (see env.py, which rewrites the workspace config), so
rollouts in the same GRPO group never see each other's tables.

Credentials: a JSON file in the same format as ELT-Bench's
setup/destination/snowflake_credential.json (account, user, password, role,
warehouse). For real training the agent-facing role should only have USAGE /
CREATE on its own two schemas; the grader uses a separate role that can read
everything. That split is configured in Snowflake, not here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from elt_rl.warehouse.base import Namespace, Warehouse


class SnowflakeWarehouse(Warehouse):
    name = "snowflake"

    def __init__(self, credential_path: str | Path, snapshot_schema: str = "AIRBYTE_SCHEMA"):
        self.creds = json.loads(Path(credential_path).read_text())
        # Schema that holds a correct, already-synced copy of the raw tables.
        # Filled once per task by running the official ELT-Bench setup; used
        # for zero-copy clones in transform-only mode.
        self.snapshot_schema = snapshot_schema
        self._conn = None

    def _connect(self):
        if self._conn is None:
            import snowflake.connector  # optional dependency

            self._conn = snowflake.connector.connect(
                account=self.creds["account"],
                user=self.creds["user"],
                password=self.creds["password"],
                role=self.creds.get("role"),
                warehouse=self.creds.get("warehouse"),
            )
        return self._conn

    @staticmethod
    def _schemas(rollout_id: str) -> tuple[str, str]:
        rid = rollout_id.upper().replace("-", "_")
        return f"R_{rid}_RAW", f"R_{rid}_MODELS"

    def create_namespace(self, rollout_id: str, database: str | None = None) -> Namespace:
        if database is None:
            raise ValueError("Snowflake namespaces need the task database")
        raw, models = self._schemas(rollout_id)
        ns = Namespace(rollout_id, f"{database}.{raw}", f"{database}.{models}", {"database": database})
        self.query(ns, f"CREATE OR REPLACE SCHEMA {ns.raw_schema}")
        self.query(ns, f"CREATE OR REPLACE SCHEMA {ns.model_schema}")
        return ns

    def drop_namespace(self, ns: Namespace) -> None:
        for schema in (ns.raw_schema, ns.model_schema):
            self.query(ns, f"DROP SCHEMA IF EXISTS {schema} CASCADE")

    def query(self, ns: Namespace, sql: str) -> pd.DataFrame:
        cur = self._connect().cursor()
        try:
            cur.execute(f"USE DATABASE {ns.handle['database']}")
            cur.execute(sql)
            if cur.description is None:
                return pd.DataFrame()
            return cur.fetch_pandas_all()
        finally:
            cur.close()

    def list_tables(self, ns: Namespace, schema: str) -> list[str]:
        db, sch = schema.split(".")
        df = self.query(
            ns,
            f"SELECT table_name FROM {db}.information_schema.tables "
            f"WHERE table_schema = '{sch.upper()}'",
        )
        return [t.lower() for t in df["TABLE_NAME"]] if not df.empty else []

    def table_size(self, ns: Namespace, schema: str, table: str) -> int:
        df = self.query(ns, f"SELECT COUNT(*) AS N FROM {schema}.{table}")
        return int(df["N"].iloc[0])

    def fetch_table(self, ns: Namespace, schema: str, table: str) -> pd.DataFrame:
        df = self.query(ns, f"SELECT * FROM {schema}.{table}")
        df.columns = [c.lower() for c in df.columns]
        return df

    def preload_raw(self, ns: Namespace, task) -> None:
        db = ns.handle["database"]
        # Zero-copy clone: near-instant and costs nothing until data changes.
        self.query(ns, f"CREATE OR REPLACE SCHEMA {ns.raw_schema} CLONE {db}.{self.snapshot_schema}")

    def copy_namespace(self, src: Namespace, rollout_id: str) -> Namespace:
        db = src.handle["database"]
        raw, models = self._schemas(rollout_id)
        dst = Namespace(rollout_id, f"{db}.{raw}", f"{db}.{models}", {"database": db})
        self.query(dst, f"CREATE OR REPLACE SCHEMA {dst.raw_schema} CLONE {src.raw_schema}")
        self.query(dst, f"CREATE OR REPLACE SCHEMA {dst.model_schema}")
        return dst

    def perturb_raw(self, ns: Namespace, tables: list[str], seed: int = 17) -> None:
        # Deterministic ~80% subset; HASH(*) keeps it stable across runs.
        for t in tables:
            self.query(
                ns,
                f"CREATE OR REPLACE TABLE {ns.raw_schema}.{t} AS SELECT * FROM {ns.raw_schema}.{t} "
                f"WHERE MOD(ABS(HASH(*)), 5) <> {seed % 5}",
            )

    def describe_for_agent(self, ns: Namespace) -> str:
        return (
            f"Destination: Snowflake. Your Airbyte connections must write into schema "
            f"`{ns.raw_schema}` (namespace already set in config.yaml) and your dbt models "
            f"must be built in schema `{ns.model_schema}`. Use the `sql` tool to inspect tables."
        )
