"""Run a single rollout and print the reward breakdown.

Scripted policy on the local fixture (no credentials):
    python scripts/rollout_once.py --policy oracle

A Tinker-sampled model on an official ELT-Bench task (needs TINKER_API_KEY,
the ELT-Bench docker services + Airbyte up, and Snowflake credentials):
    python scripts/rollout_once.py --policy model --destination snowflake \
        --eltbench-root ../ELT-Bench --task books --credential snowflake_credential.json
"""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from elt_rl.env import ELTEnvGroupBuilder, make_message_env, make_rollout, teardown
from elt_rl.policies import POLICIES, play
from elt_rl.task import load_eltbench_task, load_fixture_task
from elt_rl.warehouse import get_warehouse

ROOT = Path(__file__).resolve().parents[1]


async def scripted(args, task, warehouse, workdir):
    rollout = await make_rollout(task, warehouse, workdir / "ws", mode=args.mode)
    env = make_message_env(rollout, max_turns=args.max_turns, include_run_models=warehouse.name == "duckdb")
    try:
        actions = POLICIES[args.policy](task, transform_only=args.mode == "transform_only")
        return await play(env, actions)
    finally:
        await teardown(rollout, warehouse)


async def sampled(args, task, warehouse, workdir):
    import tinker
    from tinker_cookbook.completers import TinkerTokenCompleter
    from tinker_cookbook.rl.rollouts import do_single_rollout

    builder = ELTEnvGroupBuilder(
        task=task,
        warehouse=warehouse,
        model_name=args.model_name,
        renderer_name=None,
        group_size=1,
        workspace_root=workdir / "ws",
        mode=args.mode,
        max_turns=args.max_turns,
    )
    client = tinker.ServiceClient().create_sampling_client(base_model=args.model_name)
    policy = TinkerTokenCompleter(client, max_tokens=args.max_tokens)
    try:
        (env,) = await builder.make_envs()
        traj = await do_single_rollout(policy, env)
        last = traj.transitions[-1]
        return last.reward, last.metrics, len(traj.transitions)
    finally:
        await builder.cleanup()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="oracle", help=f"model or one of {sorted(POLICIES)}")
    p.add_argument("--destination", default="duckdb")
    p.add_argument("--task", default=str(ROOT / "fixtures" / "tiny_books"))
    p.add_argument("--eltbench-root")
    p.add_argument("--gt-dir")
    p.add_argument("--credential")
    p.add_argument("--mode", default="full", choices=["full", "transform_only"])
    p.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=2048)
    args = p.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="elt_rl_"))
    if args.destination == "duckdb":
        task = load_fixture_task(args.task)
        warehouse = get_warehouse("duckdb", root=workdir / "duckdb")
    else:
        task = load_eltbench_task(args.eltbench_root, args.task, args.destination, args.gt_dir)
        warehouse = get_warehouse(args.destination, credential_path=args.credential)

    run = sampled if args.policy == "model" else scripted
    reward, metrics, steps = asyncio.run(run(args, task, warehouse, workdir))
    print(json.dumps({"task": task.name, "policy": args.policy, "steps": steps, "reward": reward,
                      "metrics": metrics}, indent=2, default=float))


if __name__ == "__main__":
    main()
