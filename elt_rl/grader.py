"""Execution-based grading of one rollout.

Called once when the episode ends. It looks only at warehouse state and at the
pipeline files the agent left in its workspace, never at the conversation, so
the model cannot talk its way into reward.

Checks, in order:
  stage 1  every expected raw table is in the rollout's raw schema with the
           expected row count and the same content as the reference load
           (order-independent fingerprint; loader metadata columns ignored).
  stage 2  every data model matches the gold output (ELT-Bench comparator).
  rebuild  the submitted model SQL, re-executed in a fresh copy of the
           namespace, reproduces the tables the agent left behind. Tables
           typed in by hand (INSERT ... VALUES) fail this.
  perturb  (only when gold SQL is available, i.e. local fixtures) the
           submitted SQL also matches the gold SQL on a resampled copy of the
           raw data, so a pipeline that hard-codes the answer for this one
           dataset gets nothing.
"""

from __future__ import annotations

import csv
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from elt_rl.compare import compare_model
from elt_rl.task import ELTTask
from elt_rl.warehouse.base import Namespace, Warehouse, frame_fingerprint

logger = logging.getLogger(__name__)

# Reference fingerprints only depend on (warehouse, task), so compute them once
# per process instead of once per rollout.
_REF_CACHE: dict[tuple[str, str], dict[str, str]] = {}


@dataclass
class RewardWeights:
    stage1: float = 0.3
    stage2: float = 0.7
    partial_credit: float = 0.5  # max credit for a model with only some columns right


@dataclass
class ELTGrader:
    task: ELTTask
    warehouse: Warehouse
    ns: Namespace
    workspace: Path
    mode: str = "full"  # "full" (load + transform) or "transform_only"
    weights: RewardWeights = field(default_factory=RewardWeights)
    check_rebuild: bool = True
    check_perturbation: bool = True
    _ref_fingerprints: dict[str, str] | None = None

    # -- entry point used as the cookbook RewardFn -------------------------------
    async def __call__(self, history) -> tuple[float, dict[str, float]]:
        try:
            return self.grade()
        except Exception as e:  # grading must never crash the training loop
            logger.exception("grading failed for %s: %s", self.task.name, e)
            return 0.0, {"grading_error": 1.0}

    def grade(self) -> tuple[float, dict[str, float]]:
        metrics: dict[str, float] = {}

        stage1 = 1.0
        if self.mode == "full":
            stage1, s1_metrics = self._stage1()
            metrics.update(s1_metrics)

        model_scores = []
        for model in self.task.model_names:
            score, m = self._grade_model(model)
            model_scores.append(score)
            metrics.update({f"model/{model}/{k}": v for k, v in m.items()})
        stage2 = sum(model_scores) / len(model_scores) if model_scores else 0.0

        if self.mode == "full":
            reward = self.weights.stage1 * stage1 + self.weights.stage2 * stage2
        else:
            reward = stage2
        metrics.update(
            {
                "stage1_score": stage1,
                "stage2_score": stage2,
                "models_correct": float(sum(s == 1.0 for s in model_scores)),
                "models_total": float(len(model_scores)),
                "pipeline_success": float(stage1 == 1.0 and all(s == 1.0 for s in model_scores)),
            }
        )
        return reward, metrics

    # -- stage 1 ----------------------------------------------------------------
    def _stage1(self) -> tuple[float, dict[str, float]]:
        present = {t.lower() for t in self.warehouse.list_tables(self.ns, self.ns.raw_schema)}
        ref = self._reference_fingerprints()
        ok = 0
        metrics = {}
        for table, expected_rows in self.task.raw_row_counts.items():
            status = "missing"
            if table.lower() in present:
                n = self.warehouse.table_size(self.ns, self.ns.raw_schema, table)
                status = "wrong_count"
                if n == expected_rows:
                    cols = self._raw_columns(table)
                    if not cols or table not in ref:
                        status = "ok"
                    else:
                        fp = self.warehouse.fingerprint(self.ns, self.ns.raw_schema, table, cols)
                        status = "ok" if fp == ref[table] else "wrong_content"
            ok += status == "ok"
            metrics[f"raw/{table}/ok"] = float(status == "ok")
            if status == "wrong_content":
                metrics["raw_wrong_content"] = metrics.get("raw_wrong_content", 0.0) + 1
        total = len(self.task.raw_row_counts)
        return (ok / total if total else 1.0), metrics

    def _raw_columns(self, table: str) -> list[str]:
        """Source columns from schemas/<table>.csv (what the content check compares)."""
        candidates = [self.task.project_dir / "schemas" / f"{table}.csv"]
        root = self.task.extra.get("eltbench_root")
        if root:
            candidates.append(Path(root) / "elt-bench" / "schemas" / self.task.name / f"{table}.csv")
        for p in candidates:
            if p.exists():
                with p.open(newline="") as f:
                    return [row["column_name"].strip() for row in csv.DictReader(f)]
        return []

    def _reference_fingerprints(self) -> dict[str, str]:
        if self._ref_fingerprints is not None:
            return self._ref_fingerprints
        key = (self.warehouse.name, str(self.task.project_dir))
        if key in _REF_CACHE:
            self._ref_fingerprints = _REF_CACHE[key]
            return self._ref_fingerprints
        ref_ns = self.warehouse.create_namespace(f"ref_{uuid.uuid4().hex[:8]}", self._database())
        try:
            self.warehouse.preload_raw(ref_ns, self.task)
            fps = {}
            for table in self.task.raw_row_counts:
                cols = self._raw_columns(table)
                if cols:
                    fps[table] = self.warehouse.fingerprint(ref_ns, ref_ns.raw_schema, table, cols)
            self._ref_fingerprints = _REF_CACHE[key] = fps
        finally:
            self.warehouse.drop_namespace(ref_ns)
        return self._ref_fingerprints

    # -- stage 2 ----------------------------------------------------------------
    def _grade_model(self, model: str) -> tuple[float, dict[str, float]]:
        present = {t.lower() for t in self.warehouse.list_tables(self.ns, self.ns.model_schema)}
        if model.lower() not in present:
            return 0.0, {"exists": 0.0}
        out = self.warehouse.fetch_table(self.ns, self.ns.model_schema, model)
        gold = self._gold_frame(model)
        res = compare_model(gold, out, self.task.sort_keys.get(model, []))
        n_cols = len(gold.columns)
        col_frac = len(res["matched"]) / n_cols if n_cols else 0.0
        metrics = {"exists": 1.0, "col_frac": col_frac, "match": float(res["match"])}

        if res["match"]:
            score = 1.0
        elif res["row_count_ok"]:
            score = self.weights.partial_credit * col_frac
        else:
            score = 0.0

        if score > 0 and self.check_rebuild:
            if not self._rebuild_reproduces(model, out):
                metrics["rebuild_fail"] = 1.0
                score = 0.0
        if score == 1.0 and self.check_perturbation and self.task.gold_sql_dir:
            if not self._survives_perturbation(model):
                metrics["perturb_fail"] = 1.0
                score = 0.0
        return score, metrics

    def _gold_frame(self, model: str) -> pd.DataFrame:
        path = self.task.gold_csv_dir / f"{model}.csv"
        return pd.read_csv(path)

    # -- anti-hacking checks ----------------------------------------------------
    def submitted_sql(self, model: str) -> str | None:
        """The SELECT the agent submitted for a model.

        Local runs: models/<model>.sql. dbt runs: the compiled file under
        target/compiled/ (dbt resolves ref()/source() there).
        """
        plain = self.workspace / "models" / f"{model}.sql"
        if plain.exists() and "{{" not in plain.read_text():
            return plain.read_text()
        compiled = list((self.workspace / "target" / "compiled").rglob(f"{model}.sql"))
        if compiled:
            return compiled[0].read_text()
        return None

    def _run_pipeline(self, ns: Namespace, sql_source) -> set[str]:
        """Materialize every model in `ns`. Models may depend on each other, so
        retry failures a few rounds instead of requiring a declared order."""
        pending = list(self.task.model_names)
        built: set[str] = set()
        for _ in range(len(pending) + 1):
            failed = []
            for model in pending:
                sql = sql_source(model)
                if sql is None:
                    continue
                try:
                    self.warehouse.materialize(ns, model, _retarget(sql, self.ns, ns))
                    built.add(model)
                except Exception:
                    failed.append(model)
            if not failed or len(failed) == len(pending):
                break
            pending = failed
        return built

    def _rebuild_reproduces(self, model: str, agent_table: pd.DataFrame) -> bool:
        if not hasattr(self.warehouse, "copy_namespace"):
            return True
        tmp = self.warehouse.copy_namespace(self.ns, f"rebuild_{uuid.uuid4().hex[:8]}")
        try:
            for m in self.task.model_names:
                self.warehouse.query(tmp, f"DROP TABLE IF EXISTS {tmp.model_schema}.{m}")
            built = self._run_pipeline(tmp, self.submitted_sql)
            if model not in built:
                return False
            rebuilt = self.warehouse.fetch_table(tmp, tmp.model_schema, model)
            cols = list(self._gold_frame(model).columns)
            return frame_fingerprint(rebuilt, cols) == frame_fingerprint(agent_table, cols)
        finally:
            self.warehouse.drop_namespace(tmp)

    def _survives_perturbation(self, model: str) -> bool:
        if not hasattr(self.warehouse, "perturb_raw"):
            return True
        mine = self.warehouse.copy_namespace(self.ns, f"pert_a_{uuid.uuid4().hex[:8]}")
        gold = self.warehouse.copy_namespace(self.ns, f"pert_b_{uuid.uuid4().hex[:8]}")
        try:
            for ns in (mine, gold):
                for m in self.task.model_names:
                    self.warehouse.query(ns, f"DROP TABLE IF EXISTS {ns.model_schema}.{m}")
                self.warehouse.perturb_raw(ns, list(self.task.raw_row_counts), seed=17)
            self._run_pipeline(mine, self.submitted_sql)
            self._run_pipeline(gold, lambda m: (self.task.gold_sql_dir / f"{m}.sql").read_text())
            expected = self.warehouse.fetch_table(gold, gold.model_schema, model)
            try:
                got = self.warehouse.fetch_table(mine, mine.model_schema, model)
            except Exception:
                return False
            return compare_model(expected, got, self.task.sort_keys.get(model, []))["match"]
        finally:
            self.warehouse.drop_namespace(mine)
            self.warehouse.drop_namespace(gold)

    def _database(self) -> str | None:
        return self.ns.handle.get("database")


def _retarget(sql: str, src: Namespace, dst: Namespace) -> str:
    """Point schema references from one namespace to another (needed when the
    schemas carry the rollout id, as on Snowflake)."""
    for a, b in ((src.raw_schema, dst.raw_schema), (src.model_schema, dst.model_schema)):
        if a != b:
            sql = re.sub(re.escape(a), b, sql, flags=re.IGNORECASE)
    return sql
