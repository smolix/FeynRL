# FeynRL Systems and Infrastructure -- In-Depth Reference

This document covers the non-algorithmic infrastructure in FeynRL: rollout engines, weight synchronization, data pipeline, configuration, checkpointing, experiment tracking, and distributed training mechanics.

---

## Table of Contents

1. [Rollout Engines](#1-rollout-engines)
2. [Weight Synchronization](#2-weight-synchronization)
3. [Replay Buffer](#3-replay-buffer)
4. [Reward Functions](#4-reward-functions)
5. [Data Pipeline](#5-data-pipeline)
6. [Configuration System](#6-configuration-system)
7. [Checkpointing and Resume](#7-checkpointing-and-resume)
8. [Experiment Tracking](#8-experiment-tracking)
9. [Distributed Training](#9-distributed-training)
10. [Evaluation Pipeline](#10-evaluation-pipeline)

---

## 1. Rollout Engines

### 1.1 Synchronous Engine (`VLLMRolloutEngine`)

**File:** `rollouts/vllm_engine.py`

The synchronous rollout engine is a Ray remote actor that wraps a vLLM `LLM` instance for generating completions.

**Initialization:**
- Creates a vLLM `LLM` with configurable tensor parallelism, GPU memory utilization, and attention backend
- Registers `WeightSyncExtension` as the worker extension for in-place weight updates
- Seeds Python/NumPy/PyTorch RNGs for deterministic non-vLLM operations

**Generation flow (`generate()`):**

```
1. Receive list of prompts (tokenized, with optional solution for reward)
2. Set epoch-rotated seed for sampling reproducibility
3. Call vllm_engine.generate() with SamplingParams
4. For each prompt:
   a. For each of n_samples completions:
      - Extract response token IDs
      - Extract per-token logprobs from vLLM output
      - Compute reward via reward_func(prompt_data, response_data)
      - Build token-aligned and prediction-aligned tensors:
        - input_ids [T]: prompt + response concatenated
        - pred_masks [T]: 1 on prediction positions for response tokens
        - pred_dones [T]: 1 at terminal position if finish_reason == "stop"
        - pred_old_logprobs [T]: response logprobs at prediction positions
        - pred_rewards [T]: rewards at prediction positions
        - pred_zscores [T]: z-score normalized rewards
   b. Normalize rewards across the group (z-score per prompt)
   c. Compute pass@k metrics
5. Return list of sample dicts
```

**Strict on-policy mode** (`force_strict_on_policy=True`): Enforces that `temperature=1.0`, `top_p=1.0`, `top_k=-1`, no stop tokens, and `policy_version == loaded_version`. This guarantees the rollout distribution exactly matches the training policy.

**Batch-invariant mode** (`batch_invariant=True`): All engines use the same seed so the same prompt always produces the same output regardless of which engine or batch it lands in. Requires `VLLM_BATCH_INVARIANT=1` and FlashAttention backend.

### 1.2 Asynchronous Engine (`VLLMRolloutEngineAsync`)

**File:** `rollouts/vllm_engine_async.py`

Extends the synchronous engine with an asyncio event loop running in a background daemon thread. This allows non-blocking generation requests while training runs concurrently.

Used when `params.overlap.enabled=True` in the config.

### 1.3 Prediction-Aligned vs. Token-Aligned Indexing

The rollout engine produces both token-aligned and prediction-aligned tensors. The prediction-aligned versions are used by the training loop:

```
Token positions:     [0    1    2    3    4    5    6]
Token content:       [BOS  prompt...  resp0 resp1 EOS]
                                       ▲
Token-aligned mask:  [0    0    0    1     1     1  ]  (1 on response tokens)
Pred-aligned mask:   [0    0    1    1     1     0  ]  (shifted left by 1)
```

The prediction-aligned mask marks logit position `t` as valid when it predicts a response token at position `t+1`. This eliminates the need for manual shifting in the training loop.

### 1.4 Reward Normalization

Within each prompt group (all `n_samples` completions for one prompt):

```python
mean = sum(rewards) / n_samples
std = sqrt(sum((r - mean)^2) / max(n_samples - 1, 1))  # Bessel's correction
zscore_last = (reward_last - mean) / (std + eps_reward_norm)
```

If `reward_broadcast=True`, the scalar z-score is copied to all response token positions. Otherwise, only the last token receives the z-score.

For a single sample (`n_samples=1`), `mean=0, std=1` (raw reward preserved).

### 1.5 Sampling Parameters

```python
SamplingParams(
    seed=...,               # Deterministic, epoch-rotated
    n=n_samples,            # Completions per prompt
    temperature=...,        # Default 1.0 for on-policy
    top_p=..., top_k=...,
    max_tokens=...,         # Maximum response length
    logprobs=1,             # Return top-1 logprob per token
    prompt_logprobs=...,    # Optionally return prompt logprobs
    presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
)
```

---

## 2. Weight Synchronization

### 2.1 Overview

After each training epoch (or more frequently in overlap mode), updated policy weights must be transferred from the training workers to the rollout engines. FeynRL supports three methods with automatic fallback:

```
NCCL broadcast  →  Direct transfer (Ray object store)  →  Disk reload
   (fastest)            (medium)                           (slowest)
```

### 2.2 NCCL Weight Sync

**Files:** `algs/RL/common.py` (training side), `rollouts/weight_sync.py` (rollout side), `misc/nccl_utils.py` (group creation)

**Phase 1: Gather** (`gather_weights_for_nccl`)
- Called on ALL training ranks (required for ZeRO-3 collective correctness)
- Training rank 0 accumulates the full state dict via `gather_params_for_save()`
- If PEFT is active, LoRA weights are merged into base weights and parameter names are remapped to HuggingFace format
- Returns parameter metadata: `[(name, dtype_str, shape), ...]`

**Phase 2: Broadcast** (`nccl_broadcast_gathered`)
- Training rank 0 broadcasts each parameter tensor to all rollout workers
- For NCCL backend: uses `PyNcclCommunicator.broadcast()` with GPU tensors
- For Gloo backend: uses `torch.distributed.broadcast()` with CPU tensors

**Rollout side** (`WeightSyncExtension`):
- `receive_all_weights_nccl(param_metadata)`: Receives all tensors via broadcast, loads each into the vLLM model via `load_weights()`
- Each vLLM TP worker receives its slice via `collective_rpc`

**Process group creation:**
- A separate NCCL/Gloo process group is created for weight sync, independent of DeepSpeed's training group
- For NCCL: uses vLLM's `StatelessProcessGroup` + `PyNcclCommunicator` on both sides
- For Gloo: uses `torch.distributed` with a TCP init method
- Only training rank 0 + all rollout engine workers participate

### 2.3 Direct Transfer (Ray Object Store)

```python
# Training rank 0:
state_dict = gather_state_dict()  # CPU tensors

# Driver (main_rl.py):
for engine in rollout_engines:
    ray.get(engine.update_weights_direct.remote(state_dict, version))
```

The state dict is serialized to `/dev/shm` (RAM-backed tmpfs) to avoid the 4GB msgspec limit in vLLM's `collective_rpc`. Each TP worker reads from the same local file. The file is cleaned up after all workers finish.

Verification: `update_weights_direct` checks that all TP workers loaded the same number of parameters, and that at least one key matches the model's parameter names.

### 2.4 Disk-Based Reload

Fallback method: save the model to disk via `save_checkpoint()`, then call `refresh_model(model_path, version)` on each rollout engine. This destroys and recreates the vLLM engine, which is slow but always works.

### 2.5 NCCL Group Lifecycle

```
init_weight_nccl_group()  →  gather + broadcast (repeated per epoch)  →  close_weight_nccl_group()
```

The weight sync group is destroyed during `shutdown()` before the default process group to prevent stale NCCL connections.

---

## 3. Replay Buffer

**File:** `rollouts/replay_buffer.py`

The `ReplayBuffer` is a PyTorch `Dataset` that stores generated trajectories for RL training.

### 3.1 Data Structure

Each item is a dict with prediction-aligned tensors:

| Key | Shape | Description |
|-----|-------|-------------|
| `input_ids` | `[T]` | Full sequence (prompt + response) |
| `attn_masks` | `[T]` | 1 for real tokens, 0 for padding |
| `old_logps` | `[T]` | Logprobs from rollout policy (prediction-aligned) |
| `masks` | `[T]` | 1 on valid prediction positions |
| `rewards` | `[T]` | Rewards (prediction-aligned) |
| `dones` | `[T]` | 1 at terminal positions |
| `zscores` | `[T]` | Z-score normalized rewards |
| `policy_version` | `int` | Version of policy that generated this sample |

All tensors are stored on CPU. Sequences longer than `max_seq_len` are skipped (not truncated).

### 3.2 Collation

The `collate_fn` pads all tensors in a batch to `min(max_len_in_batch, max_seq_len)`:
- `input_ids` padded with `pad_token_id`
- All other tensors padded with 0

Output keys use slightly different names for the training loop: `attn_mask`, `old_logprobs`, `mask`, `done`, `zscore`.

### 3.3 Staleness Management

```python
buffer.evict_stale(min_version)  # Remove samples from old policy versions
buffer.reset()                    # Clear entire buffer between epochs
```

The `total_action_tokens` counter is maintained for token-weighted scaling and recalculated after eviction.

---

## 4. Reward Functions

**Directory:** `rewards/`

All reward functions implement the same interface:

```python
def compute_score(prompt_data: Dict, response_data: Any) -> (torch.Tensor, bool, float)
    # Returns:
    #   reward: [len(response_tokens)] tensor
    #   is_per_token: whether reward is token-level or scalar (placed on last token)
    #   correct_threshold: threshold for pass@k computation
```

### 4.1 Dummy Reward (`rewards/dummy_reward_func.py`)

Binary reward: 1.0 if `finish_reason == "stop"`, else 0.0. Placed on the last response token.

### 4.2 GSM8K Reward (`rewards/gsm8k_reward_func.py`)

Extracts the final answer from the response using the regex `####\s*(-?[0-9.,]+)` and compares it to the ground truth:
- 1.0 if the extracted answer matches exactly
- 0.0 otherwise (including when no answer is extracted)

Placed on the last response token.

### 4.3 Math Verify Reward (`rewards/math_verify_reward_func.py`)

Uses the `math_verify` library for symbolic mathematical verification:
- Parses expressions using `ExprExtractionConfig` and `LatexExtractionConfig`
- Verifies using `math_verify.grader.verify` with configurable precision
- 30-second timeout per verification

**Process pool architecture:** Verification runs in a `ProcessPoolExecutor` with `spawn` context (not `fork`) to avoid inheriting CUDA/NCCL state from the Ray actor. 8 workers by default. The pool is lazily initialized and reused across calls.

**Batch interface:** `compute_score.batch(pairs)` scores all `(prompt, response)` pairs concurrently -- all futures are submitted before any blocking, maximizing parallelism.

---

## 5. Data Pipeline

### 5.1 PairedFeed (`data_feeds/paired.py`)

Dataset for SFT training. Loads parquet files with `prompt` (chat messages) and `answer` (response text) columns.

**Tokenization:**
- Prompt is tokenized via `tokenizer.apply_chat_template(add_generation_prompt=True)`
- Answer is tokenized separately with an explicit EOS token appended
- Concatenated to form `[prompt_ids, answer_ids]`

**Loss masking:**
- Single-turn: `loss_mask` is 1 on answer tokens, 0 on prompt tokens and padding
- Multi-turn: `loss_mask` is 1 on ALL assistant responses in the conversation (not just the final answer). Ranges are computed by incrementally tokenizing the conversation to track assistant content boundaries.

**Padding/truncation:**
- Sequences are padded to `max_seq_len` with `pad_token_id`
- Long sequences are truncated to `max_seq_len` (may lose EOS token)
- Validation: at least 1 trainable token must remain after masking

### 5.2 PromptsFeed (`data_feeds/prompts.py`)

Dataset for RL rollout generation. Returns variable-length tokenized prompts without padding.

**Output:** `{prompt_token_ids: list[int], text: str, solution: str (optional)}`

Custom `collate_fn` returns a list of dicts (no tensor stacking) since prompt lengths vary.

### 5.3 PreferenceFeed (`data_feeds/preference.py`)

Dataset for DPO training. Loads parquet files with `prompt`, `answer` (chosen), and `rejected_answer` columns.

**Output:** `{input_ids: [2, T], attn_mask: [2, T], loss_mask: [2, T-1]}` where index 0 is chosen and index 1 is rejected. Both sequences are independently padded/truncated to `max_seq_len`.

### 5.4 MixedDatasetSampler (`data_feeds/mixed_sampler.py`)

Supports training on multiple datasets with configurable mixing ratios. Two modes:

**Fixed ratio per batch** (`dynamic_ratio_every_step=False`):
- Uses largest-remainder method to allocate `batch_size` samples across datasets according to `train_ratios`
- Stable rounding: if ratios are `{A: 0.7, B: 0.3}` and batch_size=10, each batch gets exactly 7 from A and 3 from B

**Dynamic ratio per step** (`dynamic_ratio_every_step=True`):
- Each step, samples are drawn from datasets using multinomial sampling with the given ratios
- More variance per step but exact expected ratios over time

Each rank uses a rank-seeded RNG for independent but reproducible sampling.

---

## 6. Configuration System

**File:** `configs/load.py`

### 6.1 Schema

Configurations use Pydantic `BaseModel` with `extra='forbid'` to catch typos:

```python
class Run(BaseModel):       # Experiment metadata, GPU allocation, timeouts
class Train(BaseModel):     # Optimizer, scheduler, algorithm hyperparams
class Model(BaseModel):     # Model paths, dtype, attention implementation
class Data(BaseModel):      # Dataset paths, tokenizer config, mixing ratios
class Rollout(BaseModel):   # vLLM generation parameters (RL only)
class Reward(BaseModel):    # Reward function config (RL only)
class Overlap(BaseModel):   # Async overlap parameters (RL only)
class Peft(BaseModel):      # LoRA configuration
class DeepSpeed(BaseModel)  # ZeRO, mixed precision, profiling
```

### 6.2 Loading and Validation

```python
config = cfg.load_and_verify(method="rl", input_yaml="config.yaml",
                              experiment_id="run_1", world_size=8, rank=0)
```

The `load_and_verify` function:
1. Parses the YAML file
2. Validates all fields against Pydantic schemas
3. Synchronizes derived fields:
   - `deepspeed.train_micro_batch_size_per_gpu = train.train_batch_size_per_gpu`
   - `deepspeed.gradient_accumulation_steps = train.gradient_accumulation_steps`
   - `deepspeed.train_batch_size = batch_per_gpu * ga_steps * world_size`
   - Optimizer and scheduler configs are auto-populated if not explicitly set
4. Returns a validated config object

### 6.3 Key Configuration Sections

**Timeouts** (for Ray operations):
| Config Key | Default | Purpose |
|-----------|---------|---------|
| `run.init_timeout` | 1800s | Training engine initialization |
| `run.rollout_timeout` | 3600s | Rollout generation per batch |
| `run.train_step_timeout` | 1800s | Single training step |
| `run.save_timeout` | 1800s | Checkpoint save |
| `run.sync_timeout` | 900s | Weight synchronization |

**NCCL Multi-Node:**
| Config Key | Purpose |
|-----------|---------|
| `run.nccl_socket_ifname` | Network interface for inter-node traffic (e.g., "eth0") |
| `run.nccl_ib_hca` | InfiniBand HCA device (e.g., "mlx5_0") |
| `run.nccl_sync_backend` | "nccl" (GPU-to-GPU) or "gloo" (CPU-based) |

---

## 7. Checkpointing and Resume

### 7.1 RL Checkpointing (`algs/RL/common.py`)

**Model checkpoints** (HuggingFace-compatible):
- `save_checkpoint(output_dir, tag)`: Saves policy and value models in safetensors format
- ZeRO-3: parameters gathered one at a time via `GatheredParameters` to avoid OOM
- PEFT: LoRA weights merged into base model (`base + (alpha/r) * lora_B @ lora_A`) and parameter names remapped to HuggingFace format
- Large models auto-sharded into 5GB chunks with `model.safetensors.index.json`
- Config and `generation_config.json` saved for vLLM compatibility

**Engine state** (optimizer, scheduler, RNG):
- `save_engine_state(dir)`: Uses DeepSpeed's native `save_checkpoint` for optimizer shards
- Stores `client_state` with Python/NumPy/PyTorch/CUDA RNG states and `_train_step_calls` counter
- `load_engine_state(dir)`: Restores all state for exact training resume

### 7.2 SFT/DPO Checkpointing (`misc/checkpoint_utils.py`)

- `save_training_checkpoint()`: Saves model weights, tokenizer, and training state
- `resume_from_checkpoint()`: Loads model + optimizer state, returns `(start_epoch, global_step)`
- Validates `CHECKPOINT_COMPLETE` marker to avoid loading incomplete checkpoints
- `cleanup_incomplete_checkpoints()`: Removes checkpoint directories missing the completion marker

### 7.3 Error-Safe Checkpointing

The `barrier_with_error_check(succeeded)` method replaces `torch.distributed.barrier()`:
```python
flag = tensor([1 if ok else 0])
all_reduce(flag, op=MIN)
if flag == 0: raise RuntimeError(...)
```

This prevents the common failure mode where one rank fails during save (e.g., disk full) while other ranks hang at the barrier until NCCL timeout.

---

## 8. Experiment Tracking

**File:** `misc/trackers.py`

### 8.1 Tracker Interface

```python
class ExperimentTracker:
    def log_metrics(self, metrics: dict, step: int)
    def log_params(self, params: dict)
    def finish()
```

### 8.2 MLflow Tracker

- Connects to an MLflow tracking server via `tracking_uri`
- Flattens nested config dicts to dot-notation keys (e.g., `train.lr`)
- Batches parameters in chunks of 100 (MLflow API limit)
- Only rank 0 initializes and logs

### 8.3 Weights & Biases Tracker

- Initializes via `wandb.init(project=..., name=experiment_id)`
- Passes `define_metric("*", step_metric="global_step")` for x-axis alignment
- Flattens config same as MLflow

---

## 9. Distributed Training

### 9.1 Ray Orchestration (RL)

**File:** `main_rl.py`, `misc/setup_rl.py`

The RL training loop is orchestrated by a Ray driver process:

```python
# 1. Create training actors (1 GPU each)
training_engines = [AlgClass.remote(...) for _ in range(training_gpus)]

# 2. Create rollout engines (TP GPUs each)
rollout_engines = [VLLMRolloutEngine.remote(...) for _ in range(num_engines)]

# 3. Training loop
for epoch in range(total_epochs):
    for prompt_batch in rollout_dataloader:
        # Shard prompts across rollout engines
        shards = shard_batch_for_engines(prompt_batch, num_engines)
        
        # Generate rollouts (parallel across engines)
        rollout_refs = [engine.generate.remote(shard, ...) for engine, shard in zip(engines, shards)]
        rollout_results = ray.get(rollout_refs)
        
        # Add to replay buffer
        replay_buffer.add_batch_seqs(merged_results)
        
        # Prepare training micro-batches
        micro_batches = prepare_training_batches(replay_buffer, ...)
        
        # Train (parallel across training ranks)
        train_refs = [engine.train_step.remote(rank, micro_batches[rank]) for rank, engine in enumerate(training_engines)]
        metrics = ray.get(train_refs)
        
    # Weight sync
    sync_weights_to_rollout_engines(...)
    
    # Save checkpoint
    save_checkpoint(...)
```

**Ray error handling** (`misc/utils.py`):
- `ray_get_with_timeout()` polls every 60s with heartbeat logging
- Detects early task failures (`RayTaskError`, `RayActorError`) to prevent collective deadlocks
- Configurable per-operation timeouts via `run.init_timeout`, `run.rollout_timeout`, etc.

### 9.2 DeepSpeed Training (All Modes)

All training uses DeepSpeed for distributed optimization:

```python
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=trainable_params,  # Only trainable for PEFT
    config=ds_config_dict
)
```

**Gradient accumulation** is managed explicitly:
```python
model_engine.set_gradient_accumulation_boundary(is_boundary)
model_engine.backward(loss)
model_engine.step()  # Only updates weights at boundaries
```

**ZeRO Stages:**
- Stage 1: Optimizer state partitioned across ranks
- Stage 2: Optimizer + gradient partitioned
- Stage 3: Optimizer + gradient + parameter partitioned (requires collective gather for forward)

### 9.3 NCCL Configuration

For multi-node setups:
```python
# Set before distributed init
os.environ["NCCL_SOCKET_IFNAME"] = config.run.nccl_socket_ifname  # e.g., "eth0"
os.environ["NCCL_IB_HCA"] = config.run.nccl_ib_hca              # e.g., "mlx5_0"
```

The `CUBLAS_WORKSPACE_CONFIG` is set for deterministic cuBLAS operations before any CUDA context setup.

---

## 10. Evaluation Pipeline

**File:** `main_eval.py`

Standalone evaluation uses the same rollout engine infrastructure but without training:

1. Load model into vLLM
2. Generate completions for test prompts
3. Compute rewards and pass@k metrics
4. Report aggregate statistics

Supports the same configuration system, reward functions, and tensor parallelism as training.

---

## Utility Reference

### `misc/utils.py`

| Function | Purpose |
|----------|---------|
| `set_random_seeds(seed, rank)` | Set all RNG seeds for determinism |
| `safe_string_to_torch_dtype(s)` | Convert "fp16"/"bfloat16"/etc. to `torch.dtype` |
| `pad_1d_to_length(x, pad_value, target_len)` | Pad or truncate 1D tensor |
| `ensure_1d(tensor, name)` | Validate tensor is 1D |
| `load_algorithm(name, registry)` | Dynamic algorithm class loading |
| `ray_get_with_timeout(refs, timeout, desc, logger)` | Robust Ray timeout with heartbeat |
| `get_experiment_dir_name(output_dir, tag, experiment_id)` | Build checkpoint path |

### `misc/metrics.py`

| Function | Purpose |
|----------|---------|
| `pass_at_k(n, c, k)` | Unbiased pass@k estimator (Chen et al. 2021) |
| `compute_pass_metrics(rewards, n_total, threshold)` | Full pass@k suite + reward statistics |

### `misc/rollout_stats.py`

| Function | Purpose |
|----------|---------|
| `new_accumulator()` | Initialize stats dict |
| `accumulate(acc, batch_stats)` | Merge batch stats |
| `summarize(acc)` | Compute final metrics: reward mean/std/min/max, truncation/EOS/stop ratios, tokens/sec |

### `misc/nccl_utils.py`

| Function | Purpose |
|----------|---------|
| `create_nccl_process_group(...)` | Create separate process group for weight broadcast |
| `NCCLBarrier` (Ray actor) | Synchronize rollout engines before broadcast |
