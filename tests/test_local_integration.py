"""Credential-free integration test.

Runs full episodes through the same message-level env that training uses
(AgentToolMessageEnv + ELTTools + ELTGrader), with a DuckDB destination and the
tiny_books fixture. A scripted oracle must get full reward; scripted cheaters
must not.
"""

import asyncio
from pathlib import Path

import pytest

from elt_rl.env import make_message_env, make_rollout, teardown
from elt_rl.policies import POLICIES, play
from elt_rl.task import load_fixture_task
from elt_rl.warehouse import DuckDBWarehouse

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_books"


def run_episode(tmp_path, policy: str, mode: str = "full"):
    async def go():
        task = load_fixture_task(FIXTURE)
        wh = DuckDBWarehouse(tmp_path / "wh")
        rollout = await make_rollout(task, wh, tmp_path / "ws", mode=mode)
        env = make_message_env(rollout, max_turns=40, include_run_models=True)
        try:
            actions = POLICIES[policy](task, transform_only=(mode == "transform_only"))
            return await play(env, actions)
        finally:
            await teardown(rollout, wh)

    reward, metrics, _ = asyncio.run(go())
    return reward, metrics


def test_oracle_gets_full_reward(tmp_path):
    reward, m = run_episode(tmp_path, "oracle")
    assert reward == pytest.approx(1.0)
    assert m["pipeline_success"] == 1.0


def test_oracle_transform_only(tmp_path):
    reward, m = run_episode(tmp_path, "oracle", mode="transform_only")
    assert reward == pytest.approx(1.0)


def test_noop_gets_nothing(tmp_path):
    reward, _ = run_episode(tmp_path, "noop")
    assert reward == 0.0


def test_partial_credit(tmp_path):
    # stage 1 complete (0.3) + one model right and one with 4/5 columns (0.5 * 0.8)
    reward, m = run_episode(tmp_path, "partial")
    assert reward == pytest.approx(0.3 + 0.7 * (1.0 + 0.4) / 2)
    assert m["models_correct"] == 1.0


def test_hand_typed_tables_fail_rebuild(tmp_path):
    reward, m = run_episode(tmp_path, "hardcode_tables")
    assert reward == pytest.approx(0.3)  # only the load counts
    assert m["model/publishers/rebuild_fail"] == 1.0


def test_hardcoded_sql_fails_perturbation(tmp_path):
    reward, m = run_episode(tmp_path, "hardcode_sql")
    assert reward == pytest.approx(0.3)
    assert m["model/publishers/perturb_fail"] == 1.0


def test_fake_raw_tables_fail_content_check(tmp_path):
    reward, m = run_episode(tmp_path, "fake_load")
    assert reward == 0.0
    assert m["stage1_score"] == 0.0


def test_cannot_read_gold_files(tmp_path):
    reward, m = run_episode(tmp_path, "read_gold")
    assert m.get("model/publishers/match", 0.0) == 0.0
    assert reward < 1.0
