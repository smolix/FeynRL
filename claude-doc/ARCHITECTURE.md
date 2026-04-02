# FeynRL Architecture Overview

FeynRL is an algorithm-first post-training framework for large language models. It supports supervised fine-tuning (SFT), preference learning (DPO), and reinforcement learning from human feedback (PPO, GRPO, CISPO, P3O). The design prioritizes algorithmic locality: implementing a new training method typically means writing a single file with its own loss and update logic, without threading changes through the orchestration, rollout, or data layers.

## High-Level System Diagram

```
                          FeynRL System Architecture
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                              Entry Points                              │
  │  main_rl.py (RL)    main_sl.py (SFT)    main_cl.py (CL)   main_eval.py│
  └───────┬───────────────────┬──────────────────┬──────────────────┬──────┘
          │                   │                  │                  │
          ▼                   ▼                  ▼                  ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │     Ray      │   │  DeepSpeed   │   │  DeepSpeed   │   │    vLLM      │
  │ Orchestrator │   │   (direct)   │   │   (direct)   │   │  (inference) │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────────────┘
         │                  │                  │
    ┌────┴────┐             │                  │
    │         │             │                  │
    ▼         ▼             ▼                  ▼
┌────────┐ ┌────────┐  ┌────────┐        ┌────────┐
│Training│ │Rollout │  │Training│        │Training│
│Workers │ │Engines │  │ Engine │        │ Engine │
│(DS+Ray)│ │(vLLM)  │  │  (DS)  │        │  (DS)  │
└───┬────┘ └───┬────┘  └───┬────┘        └───┬────┘
    │          │           │                  │
    ▼          ▼           ▼                  ▼
┌────────────────┐     ┌────────┐        ┌────────┐
│  Weight Sync   │     │  algs/ │        │  algs/ │
│ (NCCL/Direct/  │     │  SFT   │        │SFT/DPO│
│    Disk)       │     └────────┘        └────────┘
└────────────────┘
    │
    ▼
┌────────┐
│ algs/  │
│PPO/GRPO│
│CISPO/  │
│P3O     │
└────────┘
```

## Module Responsibilities

### Entry Points

| File | Purpose | Orchestration |
|------|---------|---------------|
| `main_rl.py` | RL training (PPO, GRPO, CISPO, P3O) | Ray schedules DeepSpeed training workers + vLLM rollout workers |
| `main_sl.py` | Supervised fine-tuning (SFT) | Direct DeepSpeed, no Ray |
| `main_cl.py` | Continual learning (DPO) | Direct DeepSpeed, no Ray |
| `main_eval.py` | Standalone evaluation | vLLM inference only |

The key distinction: RL requires both a training model and a separate rollout model for generation, so Ray is needed to coordinate them. SFT and DPO only use a single model (or a frozen reference), so they run directly on DeepSpeed without Ray overhead.

### `algs/` -- Algorithm Implementations

Each algorithm lives in its own subdirectory with a self-contained implementation:

```
algs/
├── RL/
│   └── common.py        # COMMON base class: shared policy gradient infrastructure
├── PPO/
│   ├── ppo.py           # Proximal Policy Optimization with GAE + value network
│   └── value_net.py     # ValueNetwork: LM backbone + scalar head
├── GRPO/
│   └── grpo.py          # Group Relative Policy Optimization (z-score advantages)
├── CISPO/
│   └── cispo.py         # Conservative In-Sample Policy Optimization
├── P3O/
│   └── p3o.py           # Proximal Policy Optimization with ESS-based clipping
├── DPO/
│   └── dpo.py           # Direct Preference Optimization
└── SFT/
    └── sft.py           # Supervised Fine-Tuning (next-token prediction)
```

**Inheritance hierarchy for RL algorithms:**

```
COMMON (algs/RL/common.py)
  ├── PPO  (Ray remote actor)
  ├── GRPO (Ray remote actor)
  ├── CISPO (Ray remote actor)
  └── P3O  (Ray remote actor)
```

SFT and DPO are standalone classes (no Ray, no COMMON base) since they run inside `main_sl.py`/`main_cl.py` directly.

**Algorithm Registry** -- algorithms are dynamically loaded at runtime via a registry dict:

```python
# main_rl.py
Algorithm_Registry = {
    'grpo': ('algs.GRPO.grpo', 'GRPO'),
    'cispo': ('algs.CISPO.cispo', 'CISPO'),
    'p3o': ('algs.P3O.p3o', 'P3O'),
    'ppo': ('algs.PPO.ppo', 'PPO'),
}

# main_sl.py
Algorithm_Registry = {
    'sft': ('algs.SFT.sft', 'SFT'),
}
```

### `rollouts/` -- Generation and Weight Synchronization

```
rollouts/
├── vllm_engine.py       # VLLMRolloutEngine: synchronous generation (Ray actor)
├── vllm_engine_async.py # VLLMRolloutEngineAsync: overlapping generation + training
├── weight_sync.py       # WeightSyncExtension: in-place weight updates on vLLM workers
└── replay_buffer.py     # ReplayBuffer: trajectory storage for RL training
```

### `rewards/` -- Pluggable Reward Functions

```
rewards/
├── dummy_reward_func.py       # Binary: 1.0 if stopped cleanly, else 0.0
├── gsm8k_reward_func.py       # Regex extraction + exact match for GSM8K
└── math_verify_reward_func.py # Symbolic verification via math_verify library
```

All reward functions expose a `compute_score(prompt_data, response_data)` interface returning `(reward_tensor, is_per_token, correct_threshold)`.

### `data_feeds/` -- Data Loading Pipeline

```
data_feeds/
├── paired.py           # PairedFeed: tokenized prompt+answer for SFT
├── prompts.py          # PromptsFeed: variable-length prompts for RL rollouts
├── preference.py       # PreferenceFeed: chosen/rejected pairs for DPO
└── mixed_sampler.py    # MixedDatasetSampler: multi-dataset mixing with ratios
```

### `configs/` -- Configuration System

```
configs/
├── load.py             # Pydantic models for validated YAML loading
└── README.md           # Full parameter reference
```

### `misc/` -- Utilities and Infrastructure

```
misc/
├── logging.py          # Distributed-aware logging (rank 0 only)
├── trackers.py         # MLflow and W&B experiment tracking
├── metrics.py          # pass@k computation
├── utils.py            # Determinism, dtype conversion, Ray timeout helpers
├── nccl_utils.py       # NCCL process group creation for weight sync
├── checkpoint_utils.py # ZeRO-3 safe checkpointing, LoRA merge
├── rollout_stats.py    # Generation statistics accumulation
└── setup_rl.py         # Ray initialization, tokenizer loading
```

## Data Flow

### RL Training Data Flow

```
┌──────────┐    tokenize     ┌──────────┐   vLLM generate   ┌──────────┐
│ Parquet  │ ──────────────► │PromptsFeed│ ────────────────► │ Rollout  │
│  Files   │                 │(prompts)  │                   │ Engine   │
└──────────┘                 └──────────┘                   └────┬─────┘
                                                                 │
                                                     samples with rewards,
                                                    logprobs, masks, dones
                                                                 │
                                                                 ▼
┌──────────┐   micro-batches  ┌──────────┐    add_batch_seqs  ┌──────────┐
│ Training │ ◄─────────────── │  Batch   │ ◄──────────────── │  Replay  │
│  Worker  │                  │ Prepare  │                    │  Buffer  │
└──────────┘                  └──────────┘                    └──────────┘
     │
     │  train_step()
     ▼
┌──────────┐
│Algorithm │  PPO: policy_loss + value_loss + GAE
│  Loss    │  GRPO: clipped ratio * z-score advantage
│Computation│  CISPO: clipped_ratio.detach() * logprob * advantage
│          │  P3O: ESS-clipped ratio * logprob * advantage
└──────────┘
     │
     │  gradient update via DeepSpeed
     ▼
┌──────────┐   NCCL broadcast   ┌──────────┐
│ Updated  │ ──────────────────► │ Rollout  │
│  Policy  │   / direct / disk   │ Engine   │
└──────────┘                     └──────────┘
```

### SFT/DPO Training Data Flow

```
┌──────────┐   tokenize    ┌───────────────┐   DataLoader    ┌──────────┐
│ Parquet  │ ────────────► │ PairedFeed    │ ──────────────► │ Training │
│  Files   │               │ (SFT) or      │                 │  Engine  │
│          │               │ PreferenceFeed│                 │(DeepSpeed│
└──────────┘               │ (DPO)        │                 └──────────┘
                           └───────────────┘                      │
                                                    forward + loss + backward
                                                                  ▼
                                                           ┌──────────┐
                                                           │ Checkpoint│
                                                           │   Save    │
                                                           └──────────┘
```

## Distributed Architecture

### RL: Ray + DeepSpeed + vLLM

In RL mode, three distributed systems work together:

1. **Ray** orchestrates the overall training loop from a driver process. It schedules:
   - N **training actors** (1 GPU each) running DeepSpeed for gradient computation
   - M **rollout actors** (TP GPUs each) running vLLM for generation

2. **DeepSpeed** handles distributed training within the training actors:
   - ZeRO Stage 1/2/3 for memory optimization
   - Gradient accumulation and synchronization
   - Each Ray training actor is one DeepSpeed rank

3. **vLLM** handles fast inference within rollout actors:
   - Tensor parallelism across GPUs within each rollout engine
   - `WeightSyncExtension` enables in-place weight updates

**Weight synchronization** uses a three-tier fallback:

```
NCCL broadcast (fastest, GPU-to-GPU)
    │ fails
    ▼
Direct transfer via Ray object store (CPU state_dict)
    │ fails
    ▼
Disk-based checkpoint reload (safetensors save/load)
```

### SFT/DPO: DeepSpeed Only

For SFT and DPO, the architecture is simpler -- `torchrun` launches multiple processes, each initializing a DeepSpeed engine. No Ray, no rollout workers. Standard `DistributedDataParallel` semantics via DeepSpeed.

## Configuration System

The configuration system uses Pydantic `BaseModel` classes with `extra='forbid'` to catch typos at load time. Configurations are loaded from YAML files and validated against strongly-typed schemas:

```python
class Config:
    run: Run            # Experiment metadata, timeouts, GPU allocation
    train: Train        # Optimizer, scheduler, algorithm hyperparameters
    model: Model        # HuggingFace model paths, dtype, attention
    data: Data          # Parquet paths, tokenizer settings, mixing ratios
    rollout: Rollout    # vLLM generation parameters (RL only)
    reward: Reward      # Reward function configuration (RL only)
    overlap: Overlap    # Async training-rollout overlap (RL only)
    peft: Peft          # LoRA configuration
    deepspeed: DeepSpeed # ZeRO optimization, mixed precision
```

DeepSpeed config fields are automatically synchronized from the train config to prevent mismatches (e.g., `train.gradient_accumulation_steps` is written into `deepspeed.gradient_accumulation_steps`).

## Key Design Patterns

### 1. Algorithm Locality

Each algorithm implements its own `compute_policy_loss()` method. The common interface is:

```python
# RL algorithms (Ray actors inheriting COMMON):
def train_step(self, engine_id, micro_batches) -> dict  # returns metrics
def compute_policy_loss(self, logprobs, old_logprobs, advantages, mask, entropies, ref_logprobs) -> (loss, denom, metrics)

# SFT/DPO (standalone classes):
def train_step(self, micro_batch, ...) -> dict
def compute_loss(self, logits, target_ids, loss_mask, ...) -> (loss, loss_sum, num_tokens)
```

### 2. Prediction-Aligned Tensors

FeynRL uses "prediction-aligned" indexing throughout the RL pipeline. In autoregressive models, token at position `t` is predicted by logits at position `t-1`. Rather than requiring every consumer to handle this shift, the rollout engine produces prediction-aligned tensors where `pred_masks[t-1]` corresponds to the prediction of token `t`. This eliminates off-by-one bugs in the training loop.

### 3. Global Token Normalization

Loss normalization across distributed ranks uses a two-step process:
1. Each rank counts its local valid tokens
2. `all_reduce(SUM)` produces the global token count (`ga_denom`)
3. Loss is scaled by `(ga_steps * world_size) / ga_denom` to cancel DeepSpeed's internal averaging and replace it with true per-token normalization

This ensures each token contributes equally to the gradient regardless of how sequences are distributed across ranks.

### 4. Deterministic Reproducibility

- All random seeds are controlled via `set_random_seeds(seed, rank)` which sets Python, NumPy, PyTorch CPU/CUDA, and cuBLAS workspace seeds
- Micro-batch shuffling in `train_step` uses a deterministic local RNG seeded by `(seed, engine_id, call_count)`
- vLLM supports `batch_invariant` mode where the same prompt produces the same output regardless of batch composition or engine assignment
- Checkpoint resume restores all RNG states including the `_train_step_calls` counter

### 5. ZeRO-3 Safe Operations

When using ZeRO Stage 3, model parameters are partitioned across GPUs. FeynRL handles this carefully:
- **Checkpointing**: Parameters are gathered one at a time via `deepspeed.zero.GatheredParameters` to avoid materializing the full model on every GPU simultaneously
- **Weight sync**: The `gather_state_dict()` method handles ZeRO-3 collective gathering before broadcasting to rollout engines
- **LoRA merge**: After gathering, PEFT adapter weights are merged into base weights and remapped to HuggingFace-compatible parameter names for vLLM loading

### 6. Error Propagation

The `barrier_with_error_check()` method replaces `torch.distributed.barrier()` in checkpoint operations. It uses `all_reduce(MIN)` of a success flag so that if any rank fails (e.g., disk full), all ranks detect the failure and raise immediately instead of hanging until NCCL timeout.
