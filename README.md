# elt-bench-rl

An RL-with-verifiable-rewards environment for [ELT-Bench](https://github.com/uiuc-kang-lab/ELT-Bench), built on [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook).

**Each rollout** is one agent attempt at an end-to-end ELT pipeline:
1. read the task spec;
2. load the sources into the warehouse;
3. build the data models;
4. check its work with SQL;
5. submit.

**The reward** comes from executing against the resulting warehouse state, with partial credit and checks against the obvious ways of gaming it. The design write-up is in [docs/design.md](docs/design.md).

## Layout

```
elt_rl/
  task.py         ELT-Bench task loader (elt_bench++ layout) + local fixtures
  warehouse/      destination interface; DuckDB (no credentials) and Snowflake
  tools.py        bash / read_file / write_file / sql / run_models / submit
  grader.py       stage 1 + stage 2 + rebuild + perturbation checks -> reward
  compare.py      comparator adapted from ELT-Bench evaluation/eva_stage2.py
  env.py          EnvGroupBuilder / RLDataset / RLDatasetBuilder (harbor_rl pattern)
  train.py        CLI -> tinker_cookbook.rl.train.main
  policies.py     scripted policies used by the tests
fixtures/tiny_books/   small task in ELT-Bench format, with a _gold/ folder for the grader
scripts/          rollout_once.py, make_fixture_gold.py
tests/            credential-free integration test
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install tinker-cookbook          # or pip install -e path/to/tinker-cookbook
pip install -e ".[dev]"              # add [snowflake] for the Snowflake destination
```

## 1. Credential-free local test

```bash
pytest -q tests/
```

This runs full episodes through the same `AgentToolMessageEnv` that training uses, with a DuckDB destination and the `tiny_books` fixture.
- **Oracle:** a scripted correct pipeline. It must get reward 1.0, in both full and transform-only mode.
- **Cheaters:** scripted policies must get **no** model credit when they:
  - type the answers into the tables;
  - put literal answers in the model SQL;
  - fake the raw load with the right row counts;
  - try to read the grader's gold files.
- **Partial:** a policy with one wrong column gets partial credit.

To print one episode's reward breakdown:

```bash
python scripts/rollout_once.py --policy oracle
python scripts/rollout_once.py --policy hardcode_sql
```

## 2. A rollout on an official ELT-Bench task

**Setup:** follow ELT-Bench's own setup (branch `elt_bench++`):
- start the docker sources and Airbyte;
- run `setup/destination/setup.sql` in Snowflake;
- run `write_config.py` and `elt_setup.sh`.

The ground-truth CSVs are unzipped to `evaluation/gt_snowflake/`. Put your Snowflake credentials in a JSON file in the same format as `setup/destination/snowflake_credential.json`.

```bash
export TINKER_API_KEY=...
python scripts/rollout_once.py --policy model --destination snowflake \
    --eltbench-root ../ELT-Bench --task books --credential snowflake_credential.json \
    --mode transform_only        # or --mode full to include the Airbyte stage
```

**Transform-only mode** clones `<DB>.AIRBYTE_SCHEMA` (a correct sync produced once by the setup) into the rollout's raw schema.

## 3. One RL step

```bash
python -m elt_rl.train destination=duckdb tasks=fixtures/tiny_books max_steps=1 group_size=4
python -m elt_rl.train destination=snowflake eltbench_root=../ELT-Bench tasks=books \
    credential=snowflake_credential.json mode=transform_only max_steps=1
```

The defaults are `Qwen/Qwen3.5-4B`, LoRA rank 32, 4 rollouts per group, and up to 30 turns. Logs go to the cookbook runs directory.

## Status and known gaps

- **Tested:** the local path (`pytest`).
- **Implemented but not yet run end to end:** the Snowflake destination, the sampled rollout, and the Tinker step. I don't have Snowflake or Tinker credentials set up yet.
- **Linux or macOS only for training:** `tml-renderers` (needed by the cookbook's tokenizers) has no Windows wheel.
- **`LocalSandbox` is not a security boundary.** For training, pass a container sandbox via `sandbox_factory`, for example cookbook's `ModalSandbox` or a container on ELT-Bench's docker network.
- **Stage 1 accepts any correct load,** not only one done through Airbyte. See the design doc.
- **The perturbation check needs gold SQL.** ELT-Bench ships gold outputs only, so on official tasks only the rebuild check runs.
