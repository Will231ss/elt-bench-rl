"""Regenerate fixtures/<task>/_gold/expected/*.csv from the gold SQL.

    python scripts/make_fixture_gold.py fixtures/tiny_books
"""

import json
import sys
import tempfile
from pathlib import Path

from elt_rl.task import load_fixture_task
from elt_rl.warehouse import DuckDBWarehouse


def main(task_dir: str) -> None:
    task = load_fixture_task(task_dir)
    with tempfile.TemporaryDirectory() as tmp:
        wh = DuckDBWarehouse(tmp)
        ns = wh.create_namespace("gold")
        wh.preload_raw(ns, task)
        counts = {t: wh.table_size(ns, ns.raw_schema, t) for t in task.raw_row_counts}
        assert counts == task.raw_row_counts, f"table.json out of date: {counts}"
        task.gold_csv_dir.mkdir(parents=True, exist_ok=True)
        for model in task.model_names:
            wh.materialize(ns, model, (task.gold_sql_dir / f"{model}.sql").read_text())
            df = wh.fetch_table(ns, ns.model_schema, model)
            df = df.sort_values(task.sort_keys[model]).reset_index(drop=True)
            df.to_csv(task.gold_csv_dir / f"{model}.csv", index=False)
            print(f"{model}: {len(df)} rows")
        wh.drop_namespace(ns)
    print(json.dumps(counts))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fixtures/tiny_books")
