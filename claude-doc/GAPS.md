# FeynRL Gap Analysis: Feature Comparison with Leading Frameworks

This document compares FeynRL against five leading RLHF/post-training frameworks -- **verl**, **AReaL**, **OpenRLHF**, **TRL** (HuggingFace), and **NeMo RL** (NVIDIA) -- to identify missing features, prioritize additions, and describe how they should work.

---

## 1. Feature Matrix

Legend: **Y** = supported, **P** = partial, **N** = not supported, **--** = not applicable

### 1.1 Algorithms

| Feature | FeynRL | verl | AReaL | OpenRLHF | TRL | NeMo RL |
|---------|--------|------|-------|----------|-----|---------|
| SFT | Y | Y | Y | Y | Y | Y |
| DPO | Y | Y | N | Y | Y | Y |
| PPO (actor-critic) | Y | Y | Y | Y | Y | N |
| GRPO | Y | Y | Y | Y | Y | Y |
| CISPO | Y | N | N | N | N | N |
| P3O (ESS-based) | Y | N | N | N | N | N |
| REINFORCE++ | N | Y | Y | Y | N | N |
| REINFORCE++ baseline | N | Y | Y | Y | N | N |
| RLOO | N | Y | Y | Y | N | N |
| DAPO | N | Y | Y | N | Y | Y |
| Dr.GRPO | N | Y | Y | Y | Y | N |
| SAPO | N | Y | Y | N | Y | N |
| GSPO | N | Y | Y | N | N | Y |
| ReMax | N | Y | N | N | N | N |
| VAPO | N | Y | N | N | N | N |
| PRIME | N | Y | N | N | N | N |
| PF-PPO | N | Y | N | N | N | N |
| SPPO | N | Y | N | N | N | N |
| LitePPO | N | N | Y | N | N | N |
| M2PO | N | N | Y | N | N | N |
| GDPO (multi-reward) | N | N | N | N | N | Y |
| On-policy distillation | N | N | Y | N | N | Y |
| Reward model training | N | Y | Y | Y | Y | Y |
| Adaptive KL controller | N | Y | Y | N | N | N |
| Multiple KL modes | N | Y | N | N | N | N |
| Multi-iteration PPO (mu>1) | N | N | N | N | Y | N |

### 1.2 Training Infrastructure

| Feature | FeynRL | verl | AReaL | OpenRLHF | TRL | NeMo RL |
|---------|--------|------|-------|----------|-----|---------|
| DeepSpeed ZeRO 1/2/3 | Y | N | N | Y | Y | N |
| PyTorch FSDP / FSDP2 | N | Y | Y | N | Y | Y |
| Megatron-LM backend | N | Y | Y | N | N | Y |
| Tensor parallelism (training) | N | Y | Y | Y | N | Y |
| Pipeline parallelism | N | Y | Y | N | N | Y |
| Sequence parallelism | N | Y | Y | N | N | Y |
| Context parallelism | N | Y | Y | N | N | Y |
| Expert parallelism (MoE) | N | Y | Y | P | N | Y |
| LoRA / PEFT | Y | Y | Y | Y | Y | Y |
| Gradient checkpointing | Y | Y | Y | Y | Y | Y |
| FP8 training | N | Y | Y | N | N | Y |
| Sequence packing | N | Y | Y | Y | Y | Y |
| Flash Attention 2 | Y | Y | Y | Y | Y | Y |
| Fused cross-entropy kernel | N | Y | N | N | N | N |
| Ring attention (long context) | N | N | N | Y | N | N |
| torch.compile support | N | Y | N | N | N | N |
| CPU offloading | N | Y | Y | N | Y | N |

### 1.3 Inference / Rollout

| Feature | FeynRL | verl | AReaL | OpenRLHF | TRL | NeMo RL |
|---------|--------|------|-------|----------|-----|---------|
| vLLM | Y | Y | Y | Y | Y | Y |
| SGLang | N | Y | Y | N | N | N |
| TensorRT-LLM | N | Y | N | N | N | N |
| Megatron inference | N | N | N | N | N | Y |
| FP8 inference | N | Y | N | N | N | Y |
| Speculative decoding | N | Y | N | N | N | N |
| Prefix caching | N | Y | Y | N | N | N |
| Interruptible generation | N | N | Y | N | N | N |
| Dynamic batch filtering | N | Y | Y | Y | N | N |
| vLLM sleep/wake mode | N | Y | N | Y | Y | N |
| Colocated inference+training | N | Y | N | Y | Y | N |

### 1.4 Async / Overlap Training

| Feature | FeynRL | verl | AReaL | OpenRLHF | TRL | NeMo RL |
|---------|--------|------|-------|----------|-----|---------|
| Sync training | Y | Y | Y | Y | Y | Y |
| Overlap mode (chunked async) | Y | N | N | N | N | N |
| Fully async RL (disaggregated) | N | Y | Y | Y | N | Y |
| ESS-driven sync trigger | Y | N | N | N | N | N |
| Staleness-aware training | N | N | Y | N | N | N |
| Importance sampling correction | N | N | Y | Y | Y | N |
| Decoupled PPO objective | N | N | Y | N | N | N |

### 1.5 Reward Infrastructure

| Feature | FeynRL | verl | AReaL | OpenRLHF | TRL | NeMo RL |
|---------|--------|------|-------|----------|-----|---------|
| Rule-based rewards | Y | Y | Y | Y | Y | Y |
| Learned reward model serving | N | Y | Y | Y | N | Y |
| Generative reward model | N | Y | N | N | N | N |
| Multi-reward combination | N | Y | N | N | Y | Y |
| Process reward models | N | Y | N | N | N | N |
| Remote reward server | N | Y | Y | Y | N | N |
| Code execution sandbox | N | Y | N | N | N | Y |
| Overlong penalty | N | Y | N | N | N | N |
| Batch reward computation | Y | Y | Y | Y | Y | Y |

### 1.6 Agentic / Multi-Turn RL

| Feature | FeynRL | verl | AReaL | OpenRLHF | TRL | NeMo RL |
|---------|--------|------|-------|----------|-----|---------|
| Multi-turn rollout | N | Y | Y | Y | N | Y |
| Tool-calling integration | N | Y | Y | Y | Y | Y |
| Agent loop abstraction | N | Y | Y | Y | N | Y |
| Environment interface (reset/step) | N | N | N | Y | N | Y |

### 1.7 Data Pipeline

| Feature | FeynRL | verl | AReaL | OpenRLHF | TRL | NeMo RL |
|---------|--------|------|-------|----------|-----|---------|
| Parquet datasets | Y | Y | Y | Y | Y | Y |
| Mixed-dataset sampling | Y | Y | N | N | N | N |
| Sequence packing | N | Y | Y | Y | Y | Y |
| Curriculum sampling | N | Y | N | N | N | N |
| Dynamic batching (length-aware) | N | Y | Y | N | N | N |
| Multi-modal data | N | Y | Y | N | Y | Y |
| Stateful dataloader (resumable) | N | Y | N | N | N | N |

### 1.8 Scalability & Deployment

| Feature | FeynRL | verl | AReaL | OpenRLHF | TRL | NeMo RL |
|---------|--------|------|-------|----------|-----|---------|
| Ray orchestration | Y | Y | Y | Y | N | Y |
| Slurm integration | N | Y | Y | Y | N | N |
| Kubernetes / SkyPilot | N | Y | Y | N | N | N |
| Flexible GPU allocation | P | Y | Y | Y | P | Y |
| Auto device mapping | N | Y | N | N | N | N |
| AMD ROCm support | N | Y | N | N | N | N |
| Ascend NPU support | N | Y | Y | N | N | N |
| Max demonstrated scale | ~multi-node | 671B/512 GPU | 1000+ GPU | 70B | 70B | 1000+ GPU |

### 1.9 Evaluation & Monitoring

| Feature | FeynRL | verl | AReaL | OpenRLHF | TRL | NeMo RL |
|---------|--------|------|-------|----------|-----|---------|
| MLflow | Y | Y | N | N | N | N |
| Weights & Biases | Y | Y | N | Y | Y | N |
| TensorBoard | N | Y | N | Y | Y | Y |
| Periodic validation in RL | N | Y | N | N | N | Y |
| Generation logging | N | Y | N | Y | Y | N |
| FLOPs counter | N | Y | N | N | N | N |
| Throughput metrics | N | Y | Y | N | N | Y |
| Checkpoint format conversion | N | Y | N | N | N | Y |

---

## 2. Prioritized Gap Discussion

The gaps are grouped into three tiers:

- **Tier 1 (High Impact)**: Features that most competitors have and users actively expect. Missing them limits adoption.
- **Tier 2 (Medium Impact)**: Features that differentiate leading frameworks and would significantly expand FeynRL's capabilities.
- **Tier 3 (Nice to Have)**: Advanced features found in 1-2 frameworks that serve specific use cases.

---

### Tier 1: High-Impact Gaps

#### 2.1 Sequence Packing

**Present in:** verl, AReaL, OpenRLHF, TRL, NeMo RL (all five competitors)

**What it is:** Instead of padding every sequence to `max_seq_len`, multiple shorter sequences are concatenated ("packed") into a single training example up to the max length. A position ID or segment mask ensures attention doesn't cross sequence boundaries.

**Why it matters:** Padding waste is substantial. If `max_seq_len=2048` but the average sequence is 500 tokens, 75% of compute is wasted on padding. Sequence packing eliminates this entirely, giving 2-4x throughput improvements in typical SFT/DPO workloads with variable-length data. For RL, packing the replay buffer micro-batches similarly reduces wasted FLOPs.

**How it should work in FeynRL:**

- **SFT/DPO path**: In `PairedFeed` / `PreferenceFeed`, instead of padding each sample to `max_seq_len`, a packing collator groups multiple samples into bins up to `max_seq_len`. Each packed example has:
  - `input_ids [T]`: concatenated sequences
  - `position_ids [T]`: reset to 0 at each sequence boundary
  - `loss_mask [T-1]`: respects both prompt masking and sequence boundaries
  - Attention uses document-level masking (block-diagonal) to prevent cross-sequence attention. This can be done via Flash Attention's `cu_seqlens` interface or a custom attention mask.

- **RL path**: In `ReplayBuffer.collate_fn`, pack multiple trajectories into a single example. The training loop already handles variable-length masks, so the main change is in collation. Prediction-aligned indexing must be maintained within each packed sub-sequence.

- **Configuration**: Add `train.use_packing: bool = False` to enable/disable.

#### 2.2 FSDP / FSDP2 Training Backend

**Present in:** verl (primary), AReaL, TRL, NeMo RL

**What it is:** PyTorch Fully Sharded Data Parallel is a native PyTorch alternative to DeepSpeed ZeRO. FSDP2 (the current recommended version) provides better composability with `torch.compile`, CPU offloading, and simpler configuration.

**Why it matters:**
- DeepSpeed-only limits users who prefer PyTorch-native training
- FSDP2 composes better with `torch.compile` for kernel fusion
- FSDP2 has better CPU offloading that works with gradient accumulation (DeepSpeed ZeRO-3 CPU offload has known issues with GA)
- Many cloud environments and research labs standardize on FSDP
- verl achieved highest throughput with FSDP2, not DeepSpeed

**How it should work in FeynRL:**
- Add a `train.backend: str = "deepspeed"` config option (values: `"deepspeed"`, `"fsdp"`)
- For `main_sl.py`/`main_cl.py`: Replace `deepspeed.initialize()` with `torch.distributed.fsdp.FullyShardedDataParallel` wrapping. The algorithm classes (`SFT`, `DPO`) need minor changes since they call `model_engine.backward()` and `model_engine.step()` -- with FSDP these become standard `loss.backward()` and `optimizer.step()`.
- For `main_rl.py`: The Ray actors in `algs/RL/common.py` would need an FSDP init path alongside the DeepSpeed one. The key difference: FSDP uses `torch.distributed` process groups directly rather than DeepSpeed's wrapper.
- Checkpoint save/load must handle FSDP's `StateDictType.FULL_STATE_DICT` or `SHARDED_STATE_DICT`.

#### 2.3 Reward Model Training and Serving

**Present in:** verl, AReaL, OpenRLHF, TRL, NeMo RL (all five)

**What it is:** Training a reward model from human preference data (Bradley-Terry pairwise ranking), then serving it during RL to score generated completions. This is the "RM" in RLHF -- many real-world use cases need learned reward models, not just rule-based rewards.

**Why it matters:** FeynRL currently only supports rule-based reward functions (GSM8K regex, math_verify). For general instruction following, helpfulness, safety, and other subjective qualities, a learned reward model is essential. Without this, FeynRL cannot support the full RLHF pipeline.

**How it should work in FeynRL:**

- **Reward model training**: Add a `RewardModelTrainer` (similar to SFT) that trains on preference pairs. The model architecture is the same as a causal LM but with a scalar head (similar to `ValueNetwork`). Loss: Bradley-Terry pairwise loss `L = -log(sigma(r_chosen - r_rejected))`.

- **Reward model serving**: Two options:
  1. **Colocated**: Load the RM as a Ray actor alongside rollout engines. After generation, score completions via the RM actor.
  2. **Remote**: Expose the RM as an HTTP endpoint. The rollout engine calls it via `reward_func`. This decouples RM serving from the training cluster.

- **Integration**: The reward function interface already supports arbitrary `compute_score()` callables. A `ModelReward` class would wrap the RM, accept `(prompt_data, response_data)`, tokenize, forward through the RM, and return the scalar reward.

#### 2.4 Multi-Turn Rollout and Agentic RL

**Present in:** verl, AReaL, OpenRLHF, NeMo RL

**What it is:** RL training where each "episode" is a multi-turn conversation -- the model generates, calls tools (code execution, search, calculators), receives tool outputs, and continues generating. The entire multi-turn trajectory is used for policy optimization.

**Why it matters:** This is the frontier of LLM post-training. Reasoning models (DeepSeek-R1, o1-style) and agent models need multi-turn RL. Without this, FeynRL is limited to single-turn response generation.

**How it should work in FeynRL:**

- **Agent loop abstraction**: Add an `AgentLoop` base class that users subclass:
  ```python
  class AgentLoop:
      def run(self, prompt, llm_client) -> Trajectory:
          # User implements: generate -> tool call -> generate -> ...
          # Returns full trajectory with all turns
  ```

- **Rollout engine extension**: The `VLLMRolloutEngine` needs a mode where it generates incrementally, yields control to the agent loop for tool execution, then continues. This requires either:
  1. Server-mode vLLM where the agent loop makes HTTP calls, or
  2. A coroutine-based approach where generation pauses at tool-call tokens.

- **Trajectory format**: Extend the replay buffer to handle multi-turn trajectories where rewards may be sparse (only at the end) or intermediate (per-tool-call).

- **Reward propagation**: For multi-turn trajectories, the final reward needs to be propagated back through the conversation. This can use GAE (PPO) or simply assign the final reward to the last turn (GRPO).

#### 2.5 Importance Sampling Correction

**Present in:** AReaL (decoupled PPO), OpenRLHF (TIS), TRL (TIS + MIS)

**What it is:** When using vLLM or any inference engine with optimizations (top-k sampling, temperature scaling, batch-level kernel optimizations), the generation distribution may differ slightly from the training model's distribution. Additionally, in async/overlap modes, rollouts may be generated by an older policy. Importance sampling correction adjusts for this mismatch.

**Why it matters:** Without correction, the policy gradient is biased. This matters most in:
- Overlap/async mode (FeynRL already has this)
- When vLLM's sampling differs from the training model
- When reusing old rollouts

**How it should work in FeynRL:**

- **Truncated Importance Sampling (TIS)**: After computing `ratio = pi_new / pi_old`, clip extreme ratios:
  ```python
  is_weight = torch.clamp(ratio, max=tis_cap)
  loss = -(is_weight * advantage * mask).sum()
  ```

- **Masked Importance Sampling (MIS)**: Instead of clamping, discard entire sequences where the max ratio exceeds a threshold:
  ```python
  seq_max_ratio = (ratio * mask).max(dim=1).values
  keep_mask = (seq_max_ratio < mis_threshold)
  ```

- **Configuration**: Add `train.importance_sampling: str = "none"` with options `"none"`, `"tis"`, `"mis"`.

#### 2.6 Additional Core Algorithms: RLOO, REINFORCE++, DAPO, Dr.GRPO

**Present in:** Multiple competitors (see matrix)

These are the most commonly requested algorithms across the ecosystem:

**RLOO (Leave-One-Out):**
- For each prompt with N completions, compute advantage for completion i as: `A_i = R_i - (1/(N-1)) * sum_{j != i} R_j`
- This is an unbiased advantage estimator that doesn't require a value function
- Very similar to GRPO but with a different normalization

**REINFORCE++ (and baseline variant):**
- REINFORCE with PPO tricks (clipping, KL penalty) but no critic network
- The "baseline" variant subtracts the mean reward: `A_i = R_i - mean(R)`
- Simpler than PPO, competitive for reasoning tasks (validated by NVIDIA, Mistral)

**DAPO (Decoupled Clip and Dynamic Sampling):**
- Decouples clip bounds: uses asymmetric clipping where the lower bound is removed entirely for "bad" actions (negative advantage), allowing the policy to more aggressively reduce their probability
- Dynamic sampling: filters out prompts where all completions have the same outcome (all correct or all wrong), since these provide no learning signal
- Token-level loss normalization instead of sequence-level
- Achieved strong results on AIME 2024

**Dr.GRPO:**
- Removes the local standard deviation normalization from GRPO
- Uses a constant divisor instead of per-group std
- Simpler and reportedly more stable in some settings

**How they should work in FeynRL:**
- Each algorithm follows the existing pattern: a new file under `algs/` implementing `compute_policy_loss()` and `train_step()`, inheriting from `COMMON`.
- RLOO and REINFORCE++ need a modified advantage computation in the rollout engine or replay buffer.
- DAPO needs changes to the rollout engine for dynamic filtering and to the loss for token-level normalization.
- Add entries to `Algorithm_Registry` in `main_rl.py`.

---

### Tier 2: Medium-Impact Gaps

#### 2.7 Fully Async RL Training

**Present in:** verl, AReaL, OpenRLHF, NeMo RL

**What it is:** Complete decoupling of generation and training into separate concurrent processes. The trainer continuously consumes from a replay buffer while rollout workers continuously generate and enqueue new data. This is distinct from FeynRL's overlap mode, which interleaves chunks within the same epoch.

**Why it matters:** Fully async RL can achieve 2-3x throughput improvements by eliminating all GPU idle time. The trainer never waits for generation, and rollout workers never wait for training. AReaL reports 2.77x speedup. This is increasingly the standard approach at scale.

**How it should work in FeynRL:**
- **Architecture**: Separate the training loop and rollout loop into independent Ray task groups. The replay buffer becomes a shared data structure (Ray actor) that both sides read/write.
- **Staleness control**: Add a configurable `max_staleness` parameter. Samples older than `current_version - max_staleness` are evicted. The existing `evict_stale()` method already supports this.
- **Off-policy correction**: Combine with importance sampling correction (Section 2.5) to handle the distribution mismatch from stale rollouts.
- **Weight update protocol**: When the trainer updates weights, it pushes them to rollout engines. Unlike sync mode, rollout engines don't need to finish current generation first -- they can either complete the current batch with old weights or (like AReaL) interrupt and restart with new weights.

#### 2.8 Learned Value Function Improvements

**Present in:** verl (VAPO), AReaL (decoupled PPO)

**What it is:** Improvements to PPO's value function training that significantly impact performance:
- **VAPO**: Uses importance-weighted value targets and length-adaptive GAE
- **Value pretraining**: Warm-start the value function on reward model outputs before RL begins
- **Shared backbone**: Policy and value can share the transformer backbone with separate heads

**Why it matters:** PPO's performance is highly sensitive to value function quality. Poor value estimates lead to high-variance advantage estimates and unstable training.

**How it should work in FeynRL:**
- Value pretraining: Add a `pretrain_value_network()` method that runs supervised regression on `(state, reward)` pairs before the main RL loop
- Shared backbone option: Modify `ValueNetwork` to optionally share weights with the policy model, adding only the value head

#### 2.9 Dynamic Batch Filtering

**Present in:** verl, AReaL, OpenRLHF

**What it is:** After generating completions, filter out prompts where all N completions have the same outcome (e.g., all correct or all incorrect). These provide no learning signal for GRPO-style algorithms since the z-score advantage would be 0 for all completions.

**Why it matters:** Filtering removes wasted training compute on uninformative batches and can improve training stability. DAPO and other recent algorithms depend on this.

**How it should work in FeynRL:**
- Add filtering in the rollout collection phase (in `main_rl.py` between rollout and replay buffer insertion):
  ```python
  def filter_groups(samples, n_samples, success_rate_lb=0.0, success_rate_ub=1.0):
      # Group samples by prompt
      # For each group, compute success_rate = num_correct / n_samples
      # Keep only groups where lb < success_rate < ub
  ```
- Config: `rollout.filter_groups: bool = False`, `rollout.success_rate_lb: float = 0.0`, `rollout.success_rate_ub: float = 1.0`

#### 2.10 vLLM Sleep/Wake Mode

**Present in:** verl, OpenRLHF, TRL

**What it is:** vLLM's sleep mode (`--enable-sleep-mode`) offloads model weights from GPU to CPU when the engine is idle, freeing GPU memory for training. When generation is needed, weights are loaded back ("wake").

**Why it matters:** In colocated setups where training and inference share GPUs, sleep/wake allows time-multiplexing: train while the rollout engine sleeps, then wake it for generation. This enables running larger models on fewer GPUs.

**How it should work in FeynRL:**
- Pass `enable_sleep_mode=True` to vLLM `LLM` constructor
- Call `llm.sleep()` before training and `llm.wake_up()` before generation
- Requires colocation support (Section 2.14)

#### 2.11 TensorBoard Support

**Present in:** verl, OpenRLHF, TRL, NeMo RL

**What it is:** Logging training metrics to TensorBoard format, either standalone or alongside MLflow/W&B.

**Why it matters:** TensorBoard is the most widely used training visualization tool. Many users expect it, and many HPC clusters have TensorBoard but not MLflow/W&B.

**How it should work in FeynRL:**
- Add a `TensorBoardTracker` class in `misc/trackers.py` implementing `ExperimentTracker`
- Use `torch.utils.tensorboard.SummaryWriter`
- Config: `run.logger_type: "tensorboard"` (or allow comma-separated for multiple)

#### 2.12 Periodic Validation During RL Training

**Present in:** verl, NeMo RL

**What it is:** Periodically running a validation/evaluation pass during RL training to monitor performance on held-out prompts.

**Why it matters:** RL training can be unstable. Periodic validation helps detect reward hacking, mode collapse, or quality degradation early. Without it, users only see training metrics (policy loss, KL, clip fraction) which don't directly indicate generation quality.

**How it should work in FeynRL:**
- Add `run.val_freq: int = 0` config (every N epochs; 0 = disabled)
- At validation, run the rollout engine on a held-out prompt set, compute rewards/pass@k, log to tracker
- Optionally log generated text samples for qualitative review (`run.log_val_generations: bool = False`)

#### 2.13 Checkpoint Format Conversion

**Present in:** verl, NeMo RL

**What it is:** Tools to convert between checkpoint formats (HuggingFace, FSDP, DeepSpeed, Megatron, safetensors).

**Why it matters:** Users often need to move models between frameworks. FeynRL saves in HuggingFace-compatible safetensors, which is good, but lacks tools to import from other formats.

#### 2.14 Colocated Model Placement

**Present in:** verl, OpenRLHF, TRL

**What it is:** Running all models (policy, reference, value, reward) on the same GPU set, time-multiplexed. Only one model is active at a time; others have their weights offloaded or in CPU memory.

**Why it matters:** For clusters with <= 64 GPUs, colocation is more efficient than dedicating separate GPU pools to training and inference. It eliminates the weight transfer cost entirely.

**How it should work in FeynRL:**
- Instead of separate Ray actors for training and rollout, a single "hybrid worker" manages both
- Training phase: load policy into GPU, train
- Rollout phase: load into vLLM (or use sleep/wake), generate
- The overlap engine partially achieves this, but full colocation with sleep/wake would be more memory-efficient

---

### Tier 3: Advanced / Nice-to-Have Gaps

#### 2.15 Megatron-LM Integration

**Present in:** verl, AReaL, NeMo RL

**What it is:** Using Megatron-LM for training, which provides tensor, pipeline, sequence, context, and expert parallelism. This enables training models beyond the DeepSpeed ZeRO-3 memory limit.

**Why it matters:** For models > 100B parameters, Megatron's 3D+ parallelism is more memory-efficient and faster than DeepSpeed ZeRO-3 alone. However, this is a very large engineering effort and primarily relevant for production-scale training.

#### 2.16 SGLang Inference Backend

**Present in:** verl, AReaL

**What it is:** SGLang is an alternative to vLLM with RadixAttention for efficient prefix caching, particularly good for multi-turn conversations and sampling multiple responses per prompt.

**Why it matters:** SGLang can be significantly faster than vLLM for GRPO-style workloads (N samples per prompt) due to automatic radix caching of the prompt. AReaL reports substantial throughput improvements.

**How it should work in FeynRL:**
- Add a `SGLangRolloutEngine` alongside `VLLMRolloutEngine`
- Config: `rollout.engine: str = "vllm"` (values: `"vllm"`, `"sglang"`)
- The interface is the same: generate completions, return samples with logprobs

#### 2.17 FP8 Inference and Training

**Present in:** verl (inference + experimental training), AReaL (training), NeMo RL (end-to-end)

**What it is:** 8-bit floating point for inference and/or training, halving memory and increasing throughput compared to BF16.

**Why it matters:** FP8 can nearly double inference throughput and significantly reduce training memory. NeMo RL reports end-to-end FP8 as a key feature for large-scale training.

#### 2.18 Multi-Modal Support (VLMs)

**Present in:** verl, AReaL, TRL, NeMo RL

**What it is:** Training vision-language models with image/video inputs alongside text.

**Why it matters:** VLM post-training (Qwen-VL, LLaVA, etc.) is a growing area. FeynRL's data pipeline and training loop assume text-only inputs.

#### 2.19 MoE (Mixture of Experts) Support

**Present in:** verl, AReaL, OpenRLHF (basic), NeMo RL

**What it is:** Expert parallelism, auxiliary load-balancing loss, router replay for deterministic training, and hybrid parallelism strategies for MoE models.

**Why it matters:** MoE models (DeepSeek-V3, Mixtral, Qwen-MoE) are increasingly popular. Training them efficiently requires expert parallelism and specific training techniques.

#### 2.20 Curriculum Sampling

**Present in:** verl

**What it is:** Progressively increasing problem difficulty during RL training. Easy problems first to build basic skills, then harder problems to push the frontier.

**Why it matters:** Can improve sample efficiency and final performance, especially for math/reasoning tasks where difficulty varies widely.

**How it should work in FeynRL:**
- Add an `AbstractCurriculumSampler` that wraps `MixedDatasetSampler`
- Takes a difficulty score per sample and a schedule function
- Adjusts sampling weights based on training progress

#### 2.21 Code Execution Sandbox

**Present in:** verl, NeMo RL

**What it is:** A sandboxed environment for executing code generated by the model during rollout, with the output used as reward signal.

**Why it matters:** Essential for code generation RL (training models to write correct code). The sandbox executes the generated code against test cases and returns pass/fail rewards.

#### 2.22 Overlong Penalty

**Present in:** verl

**What it is:** A reward penalty applied when generated output approaches `max_response_length`. Linear penalty from 0 at the buffer start to `penalty_factor` at max length.

**Why it matters:** Prevents the model from learning to "fill space" with repetitive or low-quality text to maximize token-level rewards. Important for reasoning tasks where verbose but incorrect solutions can receive partial rewards.

**How it should work in FeynRL:**
- Apply in the rollout engine's reward normalization:
  ```python
  if response_len > max_tokens - overlong_buffer:
      penalty = penalty_factor * (response_len - (max_tokens - overlong_buffer)) / overlong_buffer
      reward -= penalty
  ```
- Config: `reward.overlong_buffer: int = 0`, `reward.overlong_penalty: float = 0.0`

---

## 3. Recommended Implementation Priority

Based on the gap analysis, here is a prioritized roadmap:

### Phase 1: Core Competitiveness
1. **Sequence packing** -- Universal across competitors, largest throughput improvement
2. **RLOO + REINFORCE++ + Dr.GRPO** -- Most-requested algorithms, small implementation effort
3. **Reward model training** -- Completes the RLHF pipeline
4. **TensorBoard support** -- Low effort, high user expectation
5. **Dynamic batch filtering** -- Required for DAPO, improves training quality

### Phase 2: Expanded Capabilities
6. **DAPO algorithm** -- High-impact algorithm, requires filtering + token-level normalization
7. **Importance sampling correction (TIS/MIS)** -- Improves overlap mode correctness
8. **Periodic validation during RL** -- Essential for training monitoring
9. **Multi-turn rollout / agent loop** -- Opens agentic RL use cases
10. **Learned reward model serving** -- Enables general RLHF

### Phase 3: Scale and Performance
11. **FSDP2 backend** -- Alternative to DeepSpeed for PyTorch-native users
12. **Fully async RL** -- Major throughput improvement at scale
13. **vLLM sleep/wake + colocation** -- Memory efficiency for smaller clusters
14. **SGLang backend** -- Better throughput for multi-sample generation
15. **Overlong penalty** -- Training quality for reasoning tasks

### Phase 4: Production Features
16. **FP8 inference** -- Throughput improvement
17. **Megatron-LM backend** -- Needed for 100B+ models
18. **MoE support** -- Growing model family
19. **Multi-modal support** -- VLM training
20. **Code execution sandbox** -- Code generation RL
