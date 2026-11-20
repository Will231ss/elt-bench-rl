# ELT-Bench RLVR environment: design

## Goal

Turn an ELT-Bench task into an RL episode that tinker-cookbook can train on.
- **One rollout** is one agent attempt at a full pipeline. The agent reads the spec, loads the sources, writes the transformations, uses execution feedback, and submits.
- **The reward** comes from executing against the warehouse state the rollout produced, never from the transcript.

## 1. The task as an environment

- **Inputs:** the ELT-Bench project base:
  - `config.yaml` (sources, connections);
  - `data_model.yaml` (target models with per-column descriptions);
  - `schemas/`;
  - for real tasks, `documentation/` and `elt/main.tf`.

  The grader-side files (row counts from `table.json`, `sort_key.json`, gold outputs) are never copied into the workspace.
- **Prompt:**
  - a short system prompt describing the two stages, plus the tool schemas;
  - a task brief that says where this rollout's tables must go and which models to build.

  The agent is expected to read the project files with its tools rather than having them pasted in, which keeps the context small.
- **Tools:**
  - `bash`;
  - `read_file` / `write_file`;
  - `sql` (runs in the rollout's own namespace; at most 30 rows back);
  - `run_models` (DuckDB only; a small stand-in for `dbt run` that builds `models/<name>.sql`);
  - `submit`.

  On Snowflake, the agent uses Terraform/Airbyte and dbt through `bash`, as in the original benchmark.
- **Termination:** `submit` (via the tool's `should_stop`), the turn limit, the token budget / context overflow, or parse errors. All of these are handled by the cookbook's `AgentToolMessageEnv` / `EnvFromMessageEnv`.
- **State:** the environment is not stateless. Every step can change files, Airbyte objects and warehouse tables. So each rollout owns:
  - a copy of the project directory;
  - its own warehouse namespace: a DuckDB file, or a pair of Snowflake schemas `R_<id>_RAW` / `R_<id>_MODELS`;
  - a sandbox.

  The Snowflake schema names are written into the workspace's `config.yaml`, so Airbyte and dbt write there. `EnvGroupBuilder.cleanup()` drops everything. The rollouts in a GRPO group run side by side, so this isolation is required, not an optimization.

## 2. Reward

**A correct pipeline** loads every source table unchanged and produces every data model so that it matches the gold output under ELT-Bench's comparator:
- rows are sorted by the unique key;
- all gold columns must be present, and extra columns are allowed;
- integer columns must match exactly;
- fractional columns must be within 1% and to the gold column's precision;
- text is compared trimmed and case-insensitively.

**Outcome reward** (`elt_rl/grader.py`), computed once at the end of the episode:

```
full mode:      R = 0.3 * stage1 + 0.7 * mean_m score_m
transform-only: R = mean_m score_m

stage1  = fraction of raw tables present with the right row count AND the same content
score_m = 1                        if model m matches gold
        = 0.5 * (matched cols / gold cols)  if the row count is right but some columns are wrong
        = 0                        otherwise (or if an anti-hacking check fails, see section 4)
```

**Why not 0/1:** the strongest agent in the ELT-Bench paper gets 3.9% of data models right, and the open models get 0%. With a binary reward, almost every GRPO group has the same reward everywhere, so the advantage is zero and no gradient flows. Column-level partial credit and the stage-1 term give small models something to climb.

**Partial credit is capped at 0.5,** so building one model fully correctly always beats being nearly right on two.

**Process rewards:** none in this version. I thought about rewarding milestones such as `terraform apply` succeeding or `dbt run` compiling. They are cheap to verify, but they are easy to farm: an agent could repeatedly apply an empty config. If added, they should be capped below the value of one correct model and decayed over training.

**Curriculum:** the bigger lever is splitting the task. `mode=transform_only` pre-loads the raw tables, so the agent only does stage 2. This skips Airbyte entirely (minutes per sync), is much cheaper per rollout, and gives denser reward. Start there, then move to full mode.

## 3. Implementation

```
elt_rl/
  task.py        load ELT-Bench tasks (elt_bench++ layout) and local fixtures
  warehouse/     Warehouse interface + DuckDB (local) + Snowflake
  sandbox.py     LocalSandbox (cookbook SandboxInterface, for tests)
  tools.py       agent tools
  grader.py      stage 1 / stage 2 / rebuild / perturbation checks -> reward
  compare.py     comparator adapted from ELT-Bench evaluation/eva_stage2.py
  env.py         ELTEnvGroupBuilder / ELTDataset / ELTDatasetBuilder (harbor_rl pattern)
  train.py       chz CLI -> tinker_cookbook.rl.train.main
  policies.py    scripted policies (oracle + cheaters) for testing
fixtures/tiny_books   a 4-table, 2-model task in ELT-Bench layout with a _gold/ folder
```

- **Destination-specific code** lives only in `warehouse/`: namespaces, queries, preloading raw data, perturbing it, and copying a namespace. A new warehouse (Databricks, Redshift, BigQuery) is one subclass. The environment, tools and grader don't change.
- **The three requested checks:**
  1. **Credential-free local test:** `pytest tests/` runs full episodes through the same message-level env used in training (DuckDB + LocalSandbox + fixture).
  2. **Credentialed rollout on an official task:** `scripts/rollout_once.py --policy model --destination snowflake --task books ...`
  3. **One Tinker RL step:** `python -m elt_rl.train ... max_steps=1`

## 4. Reward hacking

| Loophole | What the environment does |
|---|---|
| Read the gold outputs | `_gold/` is never copied into the workspace. DuckDB connections run with `enable_external_access=false` and `allowed_directories=[workspace]`, so `read_csv('…/_gold/…')` fails. (On Snowflake the gold data is not in the warehouse at all; the agent role should only have rights on its own schemas.) |
| Type the answer into the model table (`INSERT … VALUES`) | **Rebuild check:** the submitted model SQL is re-executed in a fresh copy of the namespace and must reproduce the table the agent left behind. |
| Put the literal answer in the "pipeline" SQL | **Perturbation check:** the submitted SQL and the gold SQL are both run on a seeded ~80% resample of the raw data, and they must still agree. Hard-coded rows don't. (Needs gold SQL, so it currently runs only on local fixtures. ELT-Bench ships gold outputs, not gold SQL.) |
| Fake the load (right row count, wrong data), which the official stage-1 check would accept | Stage 1 also compares an order-independent content fingerprint over the source columns (Airbyte's `_airbyte_*` columns are ignored). |
| Copy a sibling rollout's tables | Separate namespaces. On Snowflake this needs per-rollout roles in production. Not done here. |
| Leftover state from an earlier rollout | Fresh namespace per rollout, dropped in `cleanup()`. |
| Comparator slack (1% tolerance, case-insensitive text, extra columns allowed) | Kept identical to the official evaluator so rewards mean the same as benchmark scores. Integer columns are already exact. |

**Still open:**
- **Stage-1 bypass:** loading the data with a script instead of Airbyte currently counts. The task wording suggests it shouldn't. Checking Airbyte job history, or the `_airbyte_*` metadata columns, would close this.
- **The LocalSandbox is not a security boundary.** Training should use a container sandbox.

## 5. Training efficiency

**Measured locally** (tiny_books, DuckDB, Windows laptop; one oracle episode):

| | first version | now |
|---|---|---|
| Setup (workspace + namespace) | 0.15 s | 0.15 s |
| Agent tool steps | 1.1 s | 0.2 s |
| Grading (stage 1 + 2 + rebuild + perturbation) | 6–8 s | 1.6 s |

- **Why grading dominated:** about 70 warehouse queries, each opening the DuckDB file (~90 ms each). On top of that, every rollout recomputed the reference fingerprints.
- **The fix:**
  - keep one connection per namespace;
  - cache the reference fingerprints per task;
  - CHECKPOINT before copying a namespace, otherwise rows still in the WAL are lost.
- **On Snowflake** the same structure applies, but the cost is dominated by Airbyte syncs (minutes) and warehouse round-trips.

**Done in the code:**
- transform-only mode;
- Snowflake zero-copy `CLONE` for preloading raw data and for the grader's rebuild / perturbation copies, which is near-instant and free until written;
- reference fingerprints computed once per task;
- tool outputs clipped (head and tail), so episodes stay within 32k tokens.

**Next steps:**
- **Cache stage 1 by config.** If a rollout's Airbyte configuration is semantically the same as a previously verified one (`terraform plan` output, or a normalized hash), clone the snapshot instead of running the sync again.
- **Run grading off the rollout's critical path** (the cookbook supports async rollouts and streaming minibatches), and put a semaphore on Airbyte.
- **Push fingerprints into the warehouse** (`HASH_AGG` on Snowflake) instead of pulling tables.

## Status

**Tested:**
- the local path: the oracle gets reward 1.0, and every scripted cheater above is caught (`tests/test_local_integration.py`).

**Implemented but not yet run** (no credentials on my side yet):
- the Snowflake destination;
- the sampled rollout on an official task;
- the Tinker training step.

The Tinker path also needs Linux, because `tml-renderers` has no Windows wheel.
