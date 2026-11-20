from elt_rl.warehouse.base import Namespace, Warehouse
from elt_rl.warehouse.duckdb_wh import DuckDBWarehouse


def get_warehouse(name: str, **kwargs) -> Warehouse:
    if name == "duckdb":
        return DuckDBWarehouse(**kwargs)
    if name == "snowflake":
        from elt_rl.warehouse.snowflake_wh import SnowflakeWarehouse

        return SnowflakeWarehouse(**kwargs)
    raise ValueError(f"unknown destination {name!r}")


__all__ = ["Namespace", "Warehouse", "DuckDBWarehouse", "get_warehouse"]
