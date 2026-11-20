"""Destination interface.

Everything that depends on the data warehouse lives behind this class, so the
environment, tools and grader never talk to Snowflake/DuckDB directly. Adding a
destination (Databricks, Redshift, BigQuery...) means writing one subclass.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Namespace:
    """Isolated slice of the warehouse owned by one rollout."""

    rollout_id: str
    raw_schema: str
    model_schema: str
    handle: dict = field(default_factory=dict)  # backend-specific (file path, database, ...)


class Warehouse(ABC):
    name: str = "base"

    @abstractmethod
    def create_namespace(self, rollout_id: str, database: str | None = None) -> Namespace: ...

    @abstractmethod
    def drop_namespace(self, ns: Namespace) -> None: ...

    @abstractmethod
    def query(self, ns: Namespace, sql: str) -> pd.DataFrame:
        """Run a statement inside the namespace. Returns an empty frame for DDL."""

    @abstractmethod
    def list_tables(self, ns: Namespace, schema: str) -> list[str]: ...

    @abstractmethod
    def table_size(self, ns: Namespace, schema: str, table: str) -> int: ...

    @abstractmethod
    def fetch_table(self, ns: Namespace, schema: str, table: str) -> pd.DataFrame: ...

    @abstractmethod
    def preload_raw(self, ns: Namespace, task) -> None:
        """Put the reference raw tables into ns.raw_schema (transform-only mode)."""

    @abstractmethod
    def describe_for_agent(self, ns: Namespace) -> str:
        """Short text telling the agent where its tables live."""

    def materialize(self, ns: Namespace, model: str, select_sql: str) -> None:
        """CREATE OR REPLACE TABLE <model_schema>.<model> AS <select>. Overridable."""
        self.query(ns, f"CREATE OR REPLACE TABLE {ns.model_schema}.{model} AS {select_sql}")

    def fingerprint(self, ns: Namespace, schema: str, table: str, columns: list[str]) -> str:
        """Order-independent hash over the given columns (ignores loader metadata columns).

        Default implementation pulls the table; backends can push this down
        (e.g. HASH_AGG in Snowflake) when tables get large.
        """
        df = self.fetch_table(ns, schema, table)
        return frame_fingerprint(df, columns)


def frame_fingerprint(df: pd.DataFrame, columns: list[str]) -> str:
    lower = {c.lower(): c for c in df.columns}
    cols = [lower[c.lower()] for c in columns if c.lower() in lower]
    if len(cols) != len(columns):
        return "missing-columns"
    rows = sorted(
        "|".join(_norm_cell(v) for v in row) for row in df[cols].itertuples(index=False)
    )
    return hashlib.sha1("\n".join(rows).encode()).hexdigest()


def _norm_cell(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip().lower()
