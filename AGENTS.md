FedScale + Bliss: Agent Operating Notes

Purpose
- Persist core project context for this repo so assistants can work without re‑explaining the setup each chat.
- Capture how training runs are launched, where key logic lives, and how your Bliss client‑selection and adaptive‑training paths integrate into FedScale.

How To Run
- Launch a job from repo root: `python docker/driver.py submit <config.yml>`.
  - Example configs: `benchmark/configs/speech/speech_oort.yml`, `benchmark/configs/speech/speech_pyramidfl.yml`.
  - The YAML’s `job_conf` section maps directly to `fedscale/cloud/config_parser.py` args. Most scripts access these via `self.args`.
- `docker/driver.py` reads the config, builds `executor_configs`, then starts:
  - Aggregator: `fedscale/cloud/aggregation/aggregator.py` (the FL server)
  - Executors: `fedscale/cloud/execution/executor.py` (workers that train/test)
- Logs:
  - Aggregated stdout/err goes to `<job_name>_logging` in repo root.
  - Aggregator writes structured JSON one‑liners (tags like ROUND_BEGIN, TEST_RESULT) to its log.

High‑Level Architecture
- Aggregator (server): orchestrates rounds, selection, model aggregation, logging, time simulation (non‑adaptive path), and gRPC with executors.
- Executors (workers): hold datasets, run client training/testing, and report results via gRPC.
- ClientManager: tracks all client metadata, online/offline state, time models, and pluggable selectors: random, Oort, PyramidFL, Bliss.
- ClientMetadata: per‑client static capacities and dynamic traces; provides the unified 3‑phase (download/compute/upload) time simulator.
- TorchClient / AdaptiveTorchClient: implement local training in the executor for the fixed‑steps and adaptive‑budget paths respectively.

Two Execution Paths (adaptive_training flag)
- Non‑adaptive (adaptive_training=False):
  - Aggregator simulates each candidate client’s round time (download + compute + upload) at round start, then keeps the K fastest that are active.
  - Executors train only the clients the aggregator dispatched (over‑commitment may select more candidates than K, but only K actually run).
- Adaptive (adaptive_training=True):
  - Aggregator still schedules rounds and advances a global virtual clock, but delegates per‑client timing to the executor.
  - Executors use `AdaptiveTorchClient` to actually train all sampled clients, simulate compute/upload in‑loop, and report an observed `wall_duration`.
  - Aggregator buffers all results, then keeps the K fastest successful and active clients, aggregates their updates, and registers feedback for the rest.

Key Files and Responsibilities

1) docker/driver.py
- Orchestration entrypoint. Parses YAML, prepares `executor_configs` from worker IPs/GPUs, launches aggregator and executors (directly, via SSH, or containers/K8s).
- Writes `<job_name>_logging` and a pickle with job metadata for termination.

2) fedscale/cloud/aggregation/aggregator.py
- gRPC server for control/data plane. Round lifecycle:
  1. `run()` → init server/model → `event_monitor()`.
  2. At the beginning of a round, select participants via `ClientManager.select_participants()` with over‑commitment.
  3. Non‑adaptive: simulate per‑client time with `ClientManager.get_completion_time()` (or PyramidFL override); set `clients_to_run`, `round_duration`, `virtual_client_clock`.
     Adaptive: dispatch all sampled clients; per‑client time is measured on the executor.
  4. Broadcast `UPDATE_MODEL`, then either `MODEL_TEST` (at eval interval) or `START_ROUND`.
  5. Collect client uploads. Non‑adaptive → aggregate on each completion; Adaptive → buffer until all replies; then keep the K fastest and aggregate.
  6. Advance `global_virtual_clock` by the round duration; log; repeat.
- Selection: `select_participants()` delegates to `ClientManager` for random/Oort/PyramidFL/Bliss.
- Non‑adaptive time path: `tictak_client_tasks()` computes completion times; tracks stragglers and per‑round durations.
- Adaptive path helpers: `_collect_adaptive_result()` buffers executor‑reported results; `_finalise_adaptive_round()` filters K fastest successful & active clients, aggregates them, and registers feedback for stragglers.
- Per‑client training config: `get_client_conf()` adds overrides. For adaptive runs it also injects the client’s dynamic trace and budget fields (download time, training budget, EWMA λ, min payload fraction, whether to run phase 2, etc.).
- Structured logs: emits JSON one‑liners like `ROUND_BEGIN` (planned clients, pre‑utilities, PyramidFL timings, Bliss predictions), and `TEST_RESULT`.

3) fedscale/cloud/client_manager.py
- Loads all client profiles from `args.clients_file` (pickle) at startup and registers them.
- Maintains `ClientMetadata` for every feasible client, filters by sample count bounds, tracks the online pool at a given wall‑clock time, and exposes:
  - `select_participants(num, cur_time)`: mode‑aware selection (random/Oort/PyramidFL/Bliss).
  - `register_feedback(client_id, reward, time_stamp, duration, success, **kwargs)`: feeds observations back into the selector.
  - `get_completion_time(...)`: server‑side time simulator using `ClientMetadata` when in non‑adaptive mode.
- Oort mode: wraps `thirdparty/oort/oort.py` training selector (explore/exploit split, pacer with preferred round duration, penalties for slow clients).
- PyramidFL mode: also wraps an Oort‑style sampler, but additionally computes per‑client overrides per round (e.g., dropout fraction and/or local steps), and supports time simulation with dropout.
- Bliss mode: wraps `thirdparty/bliss/bliss.py` training selector. When sampling:
  - Asks Bliss which unseen clients to predict and which seen clients to refresh; computes and sends each client’s last‑5 time‑window “slices” of dynamic traces (rates/availability/battery) relative to current time.
  - After selection, forwards pre‑training metadata and later post‑training utility/success feedback to Bliss.

4) fedscale/cloud/internal/client_metadata.py
- Per‑client static capacities (CPU/GPU FLOPS, peak throughput) and dynamic traces (48‑hour cyclic piecewise‑constant arrays for network and compute/availability).
- Unified 3‑phase time simulator:
  - `get_times_with_dropout(cur_time, batch_size, local_steps, model_size, model_amount_parameters, reduction_factor=0.5, dropout_p=0.0, augmentation_factor=3.0, clock_factor=1.0)` → `(t_comp, t_total)`.
  - Helpers for `get_download_time(...)` and `get_upload_time(...)`.
- Online‑ness: `is_active(cur_time)` based on activity/inactivity intervals over a 48‑hour cycle.

5) fedscale/cloud/execution/executor.py
- gRPC client to the aggregator. After dataset partitioning and comms init, runs `event_monitor()`:
  - Receives `UPDATE_MODEL`, `CLIENT_TRAIN`, `MODEL_TEST`, `SHUT_DOWN`.
  - For `CLIENT_TRAIN`, calls `Train()` which builds a framework‑specific client via `get_client_trainer()`:
    - PyTorch + adaptive_training=False → `TorchClient`
    - PyTorch + adaptive_training=True  → `AdaptiveTorchClient`
  - After training, sends an `UPLOAD_MODEL` with results (update_weight, utility, timings when adaptive).

6) fedscale/cloud/execution/torch_client.py (non‑adaptive local training)
- Trains for a fixed number of local steps. Computes a masked update (top‑k sparsification when PyramidFL sets `pyramidfl_dropout_p`) by reconstructing local weights = global + masked delta.
- Returns: `update_weight` (dict), `utility` (RMS loss × trained samples), `gsize` (delta L2 norm, for PyramidFL), and bookkeeping fields.

7) fedscale/cloud/execution/adaptive_torch_client.py (adaptive local training)
- Uses the client’s dynamic trace and a per‑round training budget to adapt compute and payload:
  - Phase 1: coarse “fit as many as possible” in chunks of `budget_recheck_steps`, simulating compute time with current device speed; early exit if remaining budget cannot cover compute+upload.
  - Optional leftover fixed‑step block if `run_phase_2` is False.
  - Phase 2: step‑by‑step fine trade‑off — after each SGD step, estimate the minimal upload time for a compressed payload; choose `keep_frac` = max(min_payload_frac, budget/t_full_upload); compress delta accordingly; stop when marginal utility gain ≤ 0 or budget exhausted.
  - Upload is simulated with the same 48‑h trace model.
- Returns detailed timings (`t_dl`, `t_comp`, `t_ul`, `wall_duration`), iterations per phase, and final `dropout_frac` (1 − keep_frac), alongside `update_weight` and `utility`.

8) thirdparty/oort/oort.py
- Oort training selector with explicit explore/exploit pools and a pacer that maintains a preferred round duration percentile; penalizes clients whose expected duration exceeds it; stochastic top‑K among high‑score candidates.
- PyramidFL extension (same base scoring) adds per‑client overrides (e.g., dropout/local_steps) and consumes feedback like gradient‑norm (`gsize`) and per‑client times to adjust future overrides.

9) thirdparty/bliss/bliss.py (+ encode.py, regressor.py)
- Bliss training selector:
  - Maintains a per‑client state (utility, success, static metadata, and short history of dynamic metadata windows).
  - Two regressors: `g` predicts utility for unseen clients from the latest dynamic windows + static features; `h` refreshes utility for seen clients from deltas of windows + static features + small history.
  - Sampling per round: from online unseen → request predictions via `g`; from online seen → request refresh via `h`; then sample participants weighted by the current utility over the seen pool.
  - Pacer identical in spirit to Oort: tracks utility over windows; relaxes or tightens `t_budget` by `pacer_delta` when utility stagnates or swings, and exposes `t_budget` back to the aggregator/clients.
- `encode.py` turns raw device metadata into fixed‑size numeric vectors (OS modernity, brand, model clusters, RAM, internal storage, battery, CPU/GPU/throughput). `regressor.py` wraps simple regressors (linreg / xgboost / MLP) behind a unified `.fit`/`.predict` API.

Important Args (from config_parser)
- `sample_mode`: `random` | `oort` | `pyramidfl` | `bliss`
- `adaptive_training`: False (aggregator simulates) | True (executor simulates and adapts)
- Oort/PyramidFL knobs: `t_budget`, `pacer_step`, `pacer_delta`, `exploration_*`, `round_penalty`, `overcommitment`.
- Bliss knobs: `number_clients_to_predict_utility`, `number_clients_to_refresh_utility`, train‑set sizes for g/h, `ema_alpha`, `g_model`, `h_model`, `collect_data`.
- Adaptive client knobs: `t_budget`, `budget_recheck_steps`, `ewma_lambda`, `min_payload_frac`, `run_phase_2`.
- Time‑model knobs: `clock_factor` (global multiplier), `batch_size`, `local_steps`, `model_size`, `model_amount_parameters` (inferred at runtime), plus per‑client dynamic traces from the clients pickle.

Data/Profiles
- Client profiles are loaded from `args.clients_file` (pickle). Each record contains:
  - Static: `CPU_FLOPS`, `GPU_FLOPS`, `peak_throughput`, identifiers and textual metadata (brand/model/os, RAM/internal_memory, battery specs).
  - Dynamic: piecewise‑constant traces over a 48‑hour cycle: `timestamps-livelab` with `rate` for bandwidth; `timestamps-carat` with `availability` and `batteryLevel`, plus `active` / `inactive` interval boundaries.
- Training data partitioning is driven by `data_map_file`.

Conventions/Guidelines For Changes
- Preserve the two‑path split on `adaptive_training`; do not mix server‑side and executor‑side time simulation.
- Aggregator event names come from `fedscale/cloud/commons.py`; do not change them.
- Keep `Executor.get_client_trainer(...)` routing stable (TorchClient vs AdaptiveTorchClient).
- If adding a new selection policy, wire it through `ClientManager` with the same trio of methods: `register_client`, `select_participants`, `register_feedback`. Reuse online‑set filtering and dynamic‑window extraction if needed.
- Structured log tags (`ROUND_BEGIN`, `TEST_RESULT`) are consumed downstream (e.g., notebooks); keep their shape stable or add new keys rather than renaming existing ones.

Quick Examples
- Oort, non‑adaptive (server‑side time simulation):
  `python docker/driver.py submit benchmark/configs/speech/speech_oort.yml`
- PyramidFL, non‑adaptive with per‑client dropout/local‑steps overrides:
  `python docker/driver.py submit benchmark/configs/speech/speech_pyramidfl.yml`
- Bliss, adaptive (example): duplicate a YAML, set `sample_mode: bliss` and `adaptive_training: True`; tune `t_budget`, `pacer_step`, `pacer_delta`.

Troubleshooting
- No logs under `benchmark/logs/...`: check `<job_name>_logging` at repo root for SSH/launch errors.
- gRPC idle loop: ensure the aggregator is reachable (`ps_ip`), ports open, and both aggregator/executors started.
- Mismatch in K vs. collected updates: verify `num_participants`, `overcommitment`, and online set size; in adaptive runs, only the K fastest successful clients are aggregated.
- Regressors: when using `xgboost`/`mlp` for Bliss, ensure the dependencies are installed in the runtime environment.

Where Results Go
- Aggregator emits round/test summaries to its log directory and (optionally) to Weights & Biases when `wandb_token` is set.
- Root log snapshots like `google_speech_logging`, `openimage_logging`, `stackoverflow_logging` gather end‑to‑end run output for quick inspection.
