"""DuckDB destination: one database file per rollout. Needs no credentials."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import duckdb
import pandas as pd

from elt_rl.warehouse.base import Namespace, Warehouse


class DuckDBWarehouse(Warehouse):
    name = "duckdb"

    def __init__(self, root: str | Path, raw_schema: str = "raw", model_schema: str = "models"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.raw_schema = raw_schema
        self.model_schema = model_schema
        # One open connection per namespace file. Opening the file costs ~90 ms
        # on Windows, and grading alone issues ~70 queries.
        self._cons: dict[str, tuple[str | None, duckdb.DuckDBPyConnection]] = {}

    def _connect(self, ns: Namespace) -> duckdb.DuckDBPyConnection:
        path, workdir = ns.handle["path"], ns.handle.get("workdir")
        cached = self._cons.get(path)
        if cached and cached[0] == workdir:
            return cached[1]
        self._close(path)
        con = duckdb.connect(path)
        self._cons[path] = (workdir, con)
        if workdir:
            # Relative paths in read_csv(...) resolve against the rollout's
            # workspace, and nothing outside it can be read (keeps the
            # grader's gold files out of reach).
            wd = Path(workdir).as_posix()
            con.execute(f"SET file_search_path = '{wd}'")
            con.execute(f"SET allowed_directories = ['{wd}']")
            con.execute("SET enable_external_access = false")
        return con

    def create_namespace(self, rollout_id: str, database: str | None = None) -> Namespace:
        path = self.root / f"{rollout_id}.duckdb"
        if path.exists():
            path.unlink()
        ns = Namespace(rollout_id, self.raw_schema, self.model_schema, {"path": str(path)})
        con = self._connect(ns)
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {self.raw_schema}")
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {self.model_schema}")
        return ns

    def _close(self, path: str) -> None:
        cached = self._cons.pop(path, None)
        if cached:
            cached[1].close()

    def drop_namespace(self, ns: Namespace) -> None:
        self._close(ns.handle["path"])
        p = Path(ns.handle["path"])
        for f in (p, p.with_suffix(".duckdb.wal")):
            if f.exists():
                f.unlink()

    def query(self, ns: Namespace, sql: str) -> pd.DataFrame:
        if ns.handle.get("workdir"):
            sql = _absolutize_paths(sql, Path(ns.handle["workdir"]).as_posix())
        cur = self._connect(ns).execute(sql)
        if cur.description is None:
            return pd.DataFrame()
        return cur.fetchdf()

    def list_tables(self, ns: Namespace, schema: str) -> list[str]:
        df = self.query(
            ns,
            f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}'",
        )
        return df["table_name"].tolist() if not df.empty else []

    def table_size(self, ns: Namespace, schema: str, table: str) -> int:
        return int(self.query(ns, f"SELECT COUNT(*) AS n FROM {schema}.{table}")["n"].iloc[0])

    def fetch_table(self, ns: Namespace, schema: str, table: str) -> pd.DataFrame:
        return self.query(ns, f"SELECT * FROM {schema}.{table}")

    def preload_raw(self, ns: Namespace, task) -> None:
        for spec in task.config.get("flat_files", []):
            src = (task.project_dir / spec["path"]).resolve()
            reader = "read_csv_auto" if spec.get("format", "csv") == "csv" else "read_json_auto"
            self.query(
                ns,
                f"CREATE OR REPLACE TABLE {ns.raw_schema}.{spec['table']} AS "
                f"SELECT * FROM {reader}('{src.as_posix()}')",
            )

    def copy_namespace(self, src: Namespace, rollout_id: str) -> Namespace:
        """Cheap clone of a whole namespace (used by the grader's rebuild check)."""
        dst_path = self.root / f"{rollout_id}.duckdb"
        self.query(src, "CHECKPOINT")  # flush the WAL, otherwise recent writes are missing from the copy
        self._close(src.handle["path"])  # Windows won't copy a file DuckDB holds open
        shutil.copyfile(src.handle["path"], dst_path)
        return Namespace(rollout_id, src.raw_schema, src.model_schema, {"path": str(dst_path)})

    def perturb_raw(self, ns: Namespace, tables: list[str], seed: int = 17) -> None:
        """Keep a seeded ~80% sample of every raw table."""
        for t in tables:
            self.query(
                ns,
                f"CREATE OR REPLACE TABLE {ns.raw_schema}.{t} AS SELECT * FROM {ns.raw_schema}.{t} "
                f"USING SAMPLE 80 PERCENT (bernoulli, {seed})",
            )

    def describe_for_agent(self, ns: Namespace) -> str:
        return (
            f"Destination: DuckDB. Load raw source tables into schema `{ns.raw_schema}` "
            f"and build the data models in schema `{ns.model_schema}`. "
            "Use the `sql` tool to run queries (DuckDB SQL dialect)."
        )


_READER_PATH = re.compile(r"(read_(?:csv|json|parquet)\w*\(\s*')(?![A-Za-z]:|/)([^']+)'", re.IGNORECASE)


def _absolutize_paths(sql: str, workdir: str) -> str:
    """With enable_external_access=false DuckDB only accepts absolute paths
    inside allowed_directories (relative ones are rejected even when
    file_search_path points there), so resolve them against the workspace."""
    return _READER_PATH.sub(lambda m: f"{m.group(1)}{workdir}/{m.group(2)}'", sql)
