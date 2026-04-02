# Algorithm Gap Analysis: Detailed Implementation Assessment

This document examines all 26 algorithms from the feature matrix in [GAPS.md](GAPS.md), assesses how each relates to FeynRL's existing implementations, estimates implementation difficulty, and determines how many truly new algorithms need to be added.

---

## Table of Contents

1. [Existing FeynRL Building Blocks](#1-existing-feynrl-building-blocks)
2. [Algorithm-by-Algorithm Analysis](#2-algorithm-by-algorithm-analysis)
3. [Classification Summary](#3-classification-summary)
4. [Net New Algorithm Count](#4-net-new-algorithm-count)
5. [Recommended Implementation Order](#5-recommended-implementation-order)

---

## 1. Existing FeynRL Building Blocks

Before analyzing each algorithm, it is critical to understand what FeynRL already provides, since many "missing" algorithms are small variants of existing code.

### 1.1 What's Already Implemented

| Building Block | File | What It Does |
|---------------|------|-------------|
| Clipped PPO ratio loss | `algs/GRPO/grpo.py:130-133` | `min(ratio * A, clip(ratio, 1-c_low, 1+c_high) * A)` |
| **Asymmetric clipping** | `algs/GRPO/grpo.py:53-54` | `clip_low` and `clip_high` are **already separate parameters** |
| Z-score advantage | `rollouts/vllm_engine.py:671-672` | `(r - mean) / (std + eps)` per prompt group |
| GAE advantage | `algs/PPO/ppo.py:127-213` | Full GAE with gamma, tau, done masking |
| Global advantage normalization | `algs/PPO/ppo.py:405-438` | All-reduce mean/std across ranks |
| Value network | `algs/PPO/value_net.py` | LM backbone + scalar head |
| Detached ratio weighting | `algs/CISPO/cispo.py:133-134` | `clipped_ratio.detach() * logprob * adv` |
| ESS-adaptive clipping | `algs/P3O/p3o.py:92-107` | `clip(ratio, 0, ESS)` with per-batch ESS |
| Variance-reduced KL (k3) | `algs/RL/common.py:99-118` | `log(pi/pi_ref) + pi_ref/pi - 1` |
| Entropy bonus | `algs/RL/common.py:59-62` | `Categorical(logits).entropy()` |
| Global token normalization | `algs/RL/common.py:132-158` | `loss_sum * (ga_steps * world_size) / ga_denom` |
| Logprob sanitization | `algs/RL/common.py:120-130` | NaN/Inf replacement |
| Replay buffer with staleness | `rollouts/replay_buffer.py:203-211` | `evict_stale(min_version)` |
| Reward broadcast | `rollouts/vllm_engine.py:674-675` | Scalar reward copied to all tokens |
| Multiple reward functions | `rewards/` | Pluggable `compute_score()` interface |
| Reference model forward | `algs/RL/common.py:65-97` | Frozen ref model for KL |

### 1.2 Key Insight: Most Algorithms Differ Only in Advantage Computation or Loss Formulation

The FeynRL architecture cleanly separates:
1. **Advantage computation** -- happens in the rollout engine (`normalize_rewards`) or in the algorithm (`calculate_gae`)
2. **Policy loss formulation** -- happens in `compute_policy_loss()`
3. **Loss normalization** -- happens in `train_step()` scaling before backward

Most "new" algorithms change only one of these three components while keeping the rest identical.

---

## 2. Algorithm-by-Algorithm Analysis

### Already Implemented (6 algorithms)

#### 1. SFT
**Status:** Fully implemented in `algs/SFT/sft.py`. Standard next-token cross-entropy with loss masking.

#### 2. DPO
**Status:** Fully implemented in `algs/DPO/dpo.py`. Length-normalized Bradley-Terry loss.

#### 3. PPO
**Status:** Fully implemented in `algs/PPO/ppo.py` + `value_net.py`. Actor-critic with GAE, clipped ratios, separate value network.

#### 4. GRPO
**Status:** Fully implemented in `algs/GRPO/grpo.py`. Clipped ratio loss with z-score advantages.

#### 5. CISPO
**Status:** Fully implemented in `algs/CISPO/cispo.py`. Detached clipped ratio as importance weight.

#### 6. P3O
**Status:** Fully implemented in `algs/P3O/p3o.py`. ESS-adaptive clipping bound.

---

### Config-Level Variants of GRPO (No New Algorithm File Needed)

These algorithms are GRPO with different advantage normalization or loss aggregation. They can be implemented as configuration options on the existing GRPO class or via modified advantage computation in the rollout engine.

#### 7. Dr.GRPO
**What it is:** GRPO with two changes: (a) no std division in advantage, (b) constant divisor `B * max_completion_length` for loss normalization instead of per-sequence length.

**Source code examined:** verl `core_algos.py:324-328` (advantage), TRL `grpo_trainer.py:2396-2398` (loss aggregation).

**How it maps to FeynRL:**
- Change (a): In `vllm_engine.py:normalize_rewards()`, the z-score currently divides by `(std + eps)`. For Dr.GRPO, replace with division by 1 (i.e., `advantage = r - mean`, no std). This is a one-line config branch.
- Change (b): The loss denominator in `grpo.py:train_step()` currently uses `local_denom` (actual token count) or `ga_denom` (global token count). Dr.GRPO would use `B * max_seq_len` instead. This is a ~3-line change.

**Implementation effort:** ~10 lines. Config flag: `train.advantage_norm: "zscore" | "mean_only"` and `train.loss_denom: "token_count" | "constant"`.

**Difficulty: Trivial**

#### 8. REINFORCE++ Baseline
**What it is:** GRPO with group-level mean subtraction (no std division), followed by batch-level whitening (z-score across the entire batch, not per-group).

**Source code examined:** verl `core_algos.py:536-584`, OpenRLHF `experience_maker.py:259-260`.

**How it maps to FeynRL:**
- Step 1: Same as Dr.GRPO -- subtract group mean, no std division. This happens in rollout engine.
- Step 2: After all groups are collected, compute batch-level mean and std of advantages, then normalize. This happens in `main_rl.py` after rollout collection, before training.

**Implementation effort:** ~20 lines. The group-level mean subtraction is the same change as Dr.GRPO. The batch-level whitening is a post-processing step on the replay buffer.

**Difficulty: Trivial**

#### 9. LitePPO
**What it is:** Group-level mean subtraction + batch-level std normalization. An intermediate between GRPO and Dr.GRPO.

**Source code examined:** AReaL `docs/en/algorithms/grpo_series.md`.

**How it maps to FeynRL:** Identical to REINFORCE++ baseline: `A_i = (r_i - mean_group) / std_batch`. Same implementation as #8.

**Implementation effort:** ~5 lines (same code path as REINFORCE++ baseline with a different config combination).

**Difficulty: Trivial**

#### 10. DAPO (Decoupled Clip and Dynamic Sampling)
**What it is:** GRPO with four modifications: (a) asymmetric clipping, (b) dynamic group filtering, (c) token-level loss normalization, (d) no KL penalty.

**Source code examined:** verl `core_algos.py:1340`, TRL `grpo_trainer.py:2396-2398`, Liger Kernel `grpo_loss.py`.

**How it maps to FeynRL:**
- (a) Asymmetric clipping: **Already supported.** `clip_low` and `clip_high` are separate parameters in FeynRL. Just set `clip_low=0.2, clip_high=0.28`.
- (b) Dynamic group filtering: **New.** After rollout collection, filter out prompt groups where all completions have the same reward (all correct or all incorrect). This is ~15 lines in `main_rl.py`.
- (c) Token-level loss normalization: **Already supported.** `normalize_loss=True` does global token normalization via `ga_denom`.
- (d) No KL penalty: **Already supported.** Set `kl_coeff=0`.

**Implementation effort:** ~15 lines for group filtering. Everything else is config.

**Difficulty: Easy** (the only real work is group filtering in the orchestration loop)

---

### Small Variants Requiring a New `compute_policy_loss()` or Advantage Function

These require a new algorithm file under `algs/` or a meaningful extension to the rollout engine, but reuse >90% of existing infrastructure.

#### 11. RLOO (Leave-One-Out)
**What it is:** For N completions per prompt, the advantage for completion i is: `A_i = r_i - mean(all rewards except r_i)`. Mathematically: `A_i = G/(G-1) * (r_i - mean_all)`.

**Source code examined:** verl `core_algos.py:587-636`, OpenRLHF `experience_maker.py:256-258`.

**How it maps to FeynRL:** The only change from GRPO is the advantage computation. The policy loss (clipped ratio) is identical. RLOO replaces z-score normalization with leave-one-out baseline in `vllm_engine.py:normalize_rewards()`:
```python
# Current GRPO: zscore = (r - mean) / (std + eps)
# RLOO: advantage = r * G/(G-1) - mean * G/(G-1)
# Or equivalently: advantage = r - (sum_others / (G-1))
```

**Does NOT need a new algorithm file.** This is a change in `normalize_rewards()` with a config switch.

**Implementation effort:** ~10 lines in `vllm_engine.py`.

**Difficulty: Trivial**

#### 12. REINFORCE++ (Plain)
**What it is:** Token-level discounted returns (cumulative sum of rewards with gamma discount), followed by batch-level whitening, using the standard PPO clipped loss.

**Source code examined:** verl `core_algos.py:693-729`.

**How it maps to FeynRL:** The key difference from GRPO is that advantages are computed at the token level (discounted returns), not as a single scalar broadcast across tokens. This requires:
1. Token-level return computation (backward cumulative sum of `pred_rewards`): ~10 lines
2. Batch-level whitening after all returns are computed: ~5 lines

The policy loss itself is identical to GRPO. This could be implemented as a mode in GRPO or a new thin algorithm file.

**Implementation effort:** ~20 lines.

**Difficulty: Easy**

#### 13. SAPO (Soft Adaptive Policy Optimization)
**What it is:** Replaces PPO's hard clipping with a smooth sigmoid gate: `gate(r, tau) = sigmoid(tau * (r - 1)) * (4/tau)`, with different temperatures for positive vs. negative advantages.

**Source code examined:** verl `core_algos.py:1615-1696`, TRL `grpo_trainer.py:2356-2359`.

**How it maps to FeynRL:** Requires a new `compute_policy_loss()` that replaces the `min(ratio*A, clip(ratio)*A)` with `gate(ratio, tau) * A`. The gate function itself is 1 line. The rest of the loss (entropy, KL, normalization) is identical.

**Options:**
- New file `algs/SAPO/sapo.py` inheriting COMMON, overriding only `compute_policy_loss()`
- Or add a `loss_type` switch to GRPO's `compute_policy_loss()`

**Implementation effort:** ~30 lines for the loss function + 2 new hyperparameters (`tau_pos`, `tau_neg`).

**Difficulty: Easy**

#### 14. GSPO (Group Sequence Policy Optimization)
**What it is:** Uses the geometric mean of per-token importance ratios across the sequence instead of per-token ratios. Uses a stop-gradient trick: `log_ratio_combined = log_prob - log_prob.detach() + sg[mean_log_ratio]`.

**Source code examined:** verl `core_algos.py:1538-1611`, AReaL `functional.py:50-141`.

**How it maps to FeynRL:** Requires modifying the ratio computation in `compute_policy_loss()`. Instead of per-token `ratio = exp(logprobs - old_logprobs)`, compute:
```python
mean_log_ratio = ((logprobs - old_logprobs) * mask).sum(dim=1) / mask.sum(dim=1)
log_ratio_combined = logprobs - logprobs.detach() + mean_log_ratio.detach().unsqueeze(1)
ratio = exp(log_ratio_combined)
```
The clipping and loss aggregation remain standard.

**Implementation effort:** ~25 lines modifying the ratio computation.

**Difficulty: Easy-Moderate** (the stop-gradient pattern requires care)

#### 15. ReMax
**What it is:** REINFORCE with a greedy-decode baseline: `A_i = R(y_i) - R(y_greedy)`. For each prompt, one additional greedy decode is run and its reward becomes the baseline.

**Source code examined:** verl `core_algos.py:732-765`.

**How it maps to FeynRL:** Two changes needed:
1. **Rollout engine**: Add a greedy decode pass (temperature=0, n=1) for each prompt alongside the stochastic samples. Store the greedy reward as `reward_baseline`.
2. **Advantage**: `advantage = r_i - reward_baseline` (simple subtraction, no group normalization).

**Implementation effort:** ~30 lines in rollout engine + ~10 lines for advantage.

**Difficulty: Easy-Moderate** (the extra greedy decode adds inference cost and rollout engine complexity)

---

### Moderate Extensions Requiring Meaningful New Code

#### 16. VAPO (Value-based Augmented PPO)
**What it is:** PPO with four modifications: (a) length-adaptive GAE lambda: `lambda = 1 - 1/(alpha * seq_len)`, (b) asymmetric clipping (already supported), (c) token-level loss normalization (already supported), (d) value pretraining phase + NLL auxiliary loss on correct samples.

**Source code examined:** Paper arxiv.org/abs/2504.05118. No standalone open-source implementation; described as a configuration of verl primitives.

**How it maps to FeynRL:**
- (a) Length-adaptive lambda: Modify `PPO.compute_advantages()` to compute `self.tau` per-sequence based on length. ~10 lines.
- (b) Already supported via `clip_low`/`clip_high`.
- (c) Already supported via `normalize_loss=True`.
- (d) Value pretraining: Add a `pretrain_value()` method that runs supervised regression on (state, return) pairs before RL begins. ~40 lines. NLL auxiliary loss: add CE loss on positive-reward sequences to the policy loss. ~15 lines.

**Implementation effort:** ~70 lines across PPO modifications + a new pretraining phase in `main_rl.py`.

**Difficulty: Moderate** (multiple orthogonal changes, touches PPO training loop)

#### 17. Multi-Iteration PPO (mu > 1)
**What it is:** Generate completions once, then perform multiple policy gradient updates on the same batch. The old logprobs are fixed at generation time; the importance ratio grows across iterations.

**Source code examined:** TRL `grpo_trainer.py:1110-1118`.

**How it maps to FeynRL:** This is a change to the training loop in `main_rl.py`, not to any algorithm's loss. Currently FeynRL generates rollouts and trains once per batch. With mu > 1:
1. Generate rollouts for a batch
2. Store `old_logprobs` (from generation time)
3. Run `train_step()` mu times on the same replay buffer data
4. `old_logprobs` stay fixed; the ratio naturally grows and clipping constrains it

**Implementation effort:** ~20 lines in `main_rl.py` training loop.

**Difficulty: Easy-Moderate** (simple loop change, but needs testing for stability)

#### 18. PF-PPO (Policy-Filtered PPO)
**What it is:** Before training, resample the replay buffer with reward-based importance weights. Higher-magnitude reward samples are upweighted.

**Source code examined:** verl `core_algos.py:2190-2266`.

**How it maps to FeynRL:** Add a `resample_by_reward()` method to `ReplayBuffer`:
```python
scores = [item['rewards'].sum() for item in self.items]
weights = torch.abs(torch.tensor(scores)) ** weight_pow
indices = torch.multinomial(weights, len(self.items), replacement=True)
self.items = [self.items[i] for i in indices]
```

**Implementation effort:** ~25 lines in `replay_buffer.py`.

**Difficulty: Easy**

#### 19. M2PO (Second-Moment Trust Policy Optimization)
**What it is:** Computes the second moment of log importance ratios across tokens. Masks out tokens whose squared log-ratio is too high (sorted by magnitude, progressively mask until the average second moment drops below threshold `tau_M2`).

**Source code examined:** AReaL `actor.py:368-556`, `functional.py`.

**How it maps to FeynRL:** Requires a new loss mask computation in `compute_policy_loss()`:
1. Compute `delta = old_logprobs - logprobs` (proximal log-ratio)
2. Sort `delta^2` in descending order
3. Find the cutoff where `mean(delta^2[cutoff:]) < tau_M2`
4. Mask out tokens above the cutoff

This is a self-contained modification to the loss function.

**Implementation effort:** ~40 lines for the M2 mask computation + integration into loss.

**Difficulty: Moderate** (novel concept, needs careful implementation of the sorting/thresholding)

---

### Different Training Paradigms (New Training Pipelines)

#### 20. On-Policy Distillation
**What it is:** Student generates on-policy sequences. Teacher model computes log-probabilities on those sequences. Training minimizes reverse KL: `D_KL(pi_student || pi_teacher)`. Can be combined with RL: `L = alpha * L_RL + beta * L_RKL`.

**Source code examined:** AReaL `actor.py:444-474`, NeMo RL `loss_functions.py:955-1037`.

**How it maps to FeynRL:** Requires:
1. A teacher model loaded alongside policy/ref models (similar to ref model, but used for distillation, not KL penalty).
2. Per-token RKL computation: `rkl_t = log_pi_student(t) - log_pi_teacher(t)`.
3. Joint loss: `L = rl_weight * L_GRPO + distill_weight * mean(rkl * mask)`.

Could be integrated into GRPO/PPO as an auxiliary loss with a teacher forward pass, or as a separate algorithm.

**Implementation effort:** ~60 lines (teacher forward pass + RKL loss + joint objective).

**Difficulty: Moderate** (needs a third model in the training pipeline)

#### 21. Reward Model Training
**What it is:** Train a reward model from human preference data using Bradley-Terry pairwise loss: `L = -log sigma(r_chosen - r_rejected)`. Architecturally similar to DPO's forward pass but training a separate model with a scalar head (like `ValueNetwork`).

**How it maps to FeynRL:** This is a new training pipeline (like `main_sl.py` but for RM training):
1. `RewardNetwork` class: reuse `ValueNetwork` (LM backbone + scalar head).
2. `RewardModelTrainer` class: similar to `DPO` but with Bradley-Terry loss on the scalar outputs.
3. `PreferenceFeed` already handles chosen/rejected pairs.
4. New entry point `main_rm.py`.

**Implementation effort:** ~150 lines (new entry point + trainer class). The `ValueNetwork` and `PreferenceFeed` are already reusable.

**Difficulty: Moderate** (mostly glue code, but a new pipeline)

#### 22. SPPO (Self-Play Preference Optimization)
**What it is:** Iterative self-play: current policy generates against previous policy. A preference oracle ranks responses. Policy updated to maximize win-rate. Converges to Nash equilibrium.

**Source code examined:** No implementation found in verl/AReaL/OpenRLHF (only documented as a recipe placeholder).

**How it maps to FeynRL:** This is a fundamentally different training paradigm requiring:
1. Two policy versions maintained simultaneously (current + previous)
2. A preference oracle (reward model or LLM judge)
3. An iterative loop: generate with both policies, compare, update

This is closer to a research project than a feature addition.

**Implementation effort:** ~300+ lines (new training loop, preference comparison, dual policy management).

**Difficulty: High**

---

### Infrastructure/Config Extensions (Not New Algorithms)

These items from the feature matrix are not algorithms but configuration extensions to existing infrastructure.

#### 23. Adaptive KL Controller
**What it is:** Dynamically adjusts `kl_coeff` using a proportional controller: `kl_coeff *= 1 + clip((KL_current/KL_target - 1), -0.2, 0.2) * n_steps/horizon`.

**Source code examined:** verl `core_algos.py:153-212`.

**How it maps to FeynRL:** Add a ~30-line `AdaptiveKLController` class that wraps the existing `kl_coeff`:
```python
class AdaptiveKLController:
    def __init__(self, init_coeff, target_kl, horizon):
        self.value = init_coeff
        self.target = target_kl
        self.horizon = horizon
    def update(self, current_kl, n_steps):
        error = np.clip(current_kl / self.target - 1, -0.2, 0.2)
        self.value *= 1 + error * n_steps / self.horizon
```
Called after each training step in the RL loop.

**Implementation effort:** ~30 lines.

**Difficulty: Trivial**

#### 24. Multiple KL Penalty Modes
**What it is:** Different KL estimators beyond the current variance-reduced form. Common modes:
- **k1 (simple):** `log(pi/pi_ref)` -- biased gradient but unbiased estimate
- **k2 (mse):** `0.5 * (log(pi) - log(pi_ref))^2` -- unbiased gradient
- **k3 (low-var, current):** `exp(log(pi_ref/pi)) - log(pi_ref/pi) - 1` -- low variance
- **abs:** `|log(pi) - log(pi_ref)|`
- **k3+ (straight-through):** forward uses k3 value, backward uses k2 gradient

**Source code examined:** verl `core_algos.py:2126-2187`.

**How it maps to FeynRL:** Extend `compute_kl_distance()` with a `kl_mode` parameter. FeynRL currently hardcodes k3. Adding the others is ~40 lines of branching.

**Implementation effort:** ~40 lines in `common.py`.

**Difficulty: Trivial**

#### 25. GDPO (Multi-Reward Normalization)
**What it is:** When multiple reward signals exist (e.g., correctness + format + length), normalize each dimension independently within its group, then aggregate with configurable weights.

**Source code examined:** verl `core_algos.py:380-468`, NeMo RL `advantage_estimator.py:77-147`.

**How it maps to FeynRL:** Currently FeynRL's reward functions return a single scalar. To support GDPO:
1. Extend the reward interface to return multiple named reward components.
2. In `normalize_rewards()`, normalize each component independently.
3. Aggregate: `advantage = sum(w_k * (r_k - mean_k) / (std_k + eps))`.
4. Optionally whiten the aggregated advantage.

**Implementation effort:** ~50 lines (reward interface change + multi-dimensional normalization).

**Difficulty: Easy-Moderate** (requires reward interface extension)

#### 26. PRIME
**What it is:** Not a training algorithm. It is a reward manager that uses a process reward model to evaluate mathematical reasoning steps. The training itself uses standard GRPO/PPO.

**Source code examined:** verl `workers/reward_manager/prime.py`.

**How it maps to FeynRL:** This would be a new reward function in `rewards/`, not a new algorithm. The existing `compute_score()` interface already supports this -- just implement a reward function that calls a process reward model.

**Implementation effort:** Depends on the PRM serving infrastructure. The training algorithm is unchanged.

**Difficulty: N/A for algorithm gap analysis** (infrastructure concern)

---

## 3. Classification Summary

### Category A: Already Implemented (6)

| # | Algorithm | Status |
|---|-----------|--------|
| 1 | SFT | Fully implemented |
| 2 | DPO | Fully implemented |
| 3 | PPO | Fully implemented |
| 4 | GRPO | Fully implemented |
| 5 | CISPO | Fully implemented |
| 6 | P3O | Fully implemented |

### Category B: Config-Level Variants of GRPO (4)

These need only changes to advantage computation or loss normalization, with no new algorithm file.

| # | Algorithm | What Changes | Lines of Code |
|---|-----------|-------------|---------------|
| 7 | Dr.GRPO | Remove std from advantage + constant loss denom | ~10 |
| 8 | REINFORCE++ baseline | Group mean (no std) + batch whitening | ~20 |
| 9 | LitePPO | Group mean + batch std | ~5 (same path as #8) |
| 10 | DAPO | Group filtering + config flags | ~15 (filtering only; rest is config) |

### Category C: Small New Loss/Advantage Functions (7)

These reuse >90% of existing COMMON infrastructure but need a new `compute_policy_loss()` or advantage function.

| # | Algorithm | What's New | Lines of Code | Could Share File |
|---|-----------|-----------|---------------|-----------------|
| 11 | RLOO | Leave-one-out baseline | ~10 | Rollout engine change |
| 12 | REINFORCE++ | Token-level discounted returns + batch whitening | ~20 | New advantage mode |
| 13 | SAPO | Sigmoid gate replaces hard clipping | ~30 | New loss or GRPO mode |
| 14 | GSPO | Sequence-level geometric mean ratio | ~25 | New loss variant |
| 15 | ReMax | Greedy baseline + extra decode | ~40 | Rollout + advantage |
| 17 | Multi-iter PPO (mu>1) | Training loop repeats on same data | ~20 | main_rl.py change |
| 18 | PF-PPO | Reward-weighted replay resampling | ~25 | Replay buffer addition |

### Category D: Moderate Extensions to PPO (2)

| # | Algorithm | What's New | Lines of Code |
|---|-----------|-----------|---------------|
| 16 | VAPO | Length-adaptive GAE + value pretraining + NLL aux loss | ~70 |
| 19 | M2PO | Second-moment token masking for off-policy stability | ~40 |

### Category E: New Training Paradigms (2)

| # | Algorithm | What's New | Lines of Code |
|---|-----------|-----------|---------------|
| 20 | On-policy distillation | Teacher model + RKL loss | ~60 |
| 21 | Reward model training | New pipeline: RM training with BT loss | ~150 |

### Category F: Significantly New System (1)

| # | Algorithm | What's New | Lines of Code |
|---|-----------|-----------|---------------|
| 22 | SPPO | Iterative self-play with preference oracle | ~300+ |

### Category G: Infrastructure Extensions, Not Algorithms (4)

| # | Item | What's New | Lines of Code |
|---|------|-----------|---------------|
| 23 | Adaptive KL controller | Proportional controller for kl_coeff | ~30 |
| 24 | Multiple KL modes | k1/k2/k3/abs/k3+ estimators | ~40 |
| 25 | GDPO | Multi-reward normalization | ~50 |
| 26 | PRIME | Process reward manager | N/A (reward function) |

---

## 4. Net New Algorithm Count

### How many new algorithm files under `algs/` are actually needed?

**Answer: 2-5 new files**, depending on implementation strategy.

The key insight is that most "algorithms" in the ecosystem are **not fundamentally different algorithms** -- they are combinations of:
1. A specific advantage estimator (z-score, RLOO, greedy baseline, GAE, token-level returns)
2. A specific loss formulation (PPO clip, detached weight, sigmoid gate, sequence-level ratio)
3. A specific normalization mode (per-group, batch-level, token-level, constant divisor)

FeynRL could implement all 20 missing items (excluding SPPO) through a combination of:

**Strategy A: Minimal new files (recommended)**

| New File | Algorithms Covered |
|----------|-------------------|
| None needed | Dr.GRPO, RLOO, REINFORCE++ baseline, LitePPO, DAPO, REINFORCE++, PF-PPO, Multi-iter |
| `algs/SAPO/sapo.py` | SAPO |
| `algs/GSPO/gspo.py` | GSPO |
| Extend `algs/PPO/ppo.py` | VAPO (config flags on PPO) |
| Extend `algs/RL/common.py` | M2PO (as a mixin/mode) |
| `main_rm.py` + `algs/RM/rm.py` | Reward model training |

This means **2 new algorithm files** (SAPO, GSPO) + 1 new entry point (RM training) + extensions to existing files.

The following 12 algorithms are **config-level variants** that need no new algorithm file:
- Dr.GRPO, RLOO, REINFORCE++, REINFORCE++ baseline, LitePPO, DAPO, ReMax, PF-PPO, Multi-iter PPO, Adaptive KL, Multiple KL modes, GDPO

**Strategy B: One file per algorithm (maximum clarity)**

If each algorithm gets its own file for clarity and discoverability:

| New File | Difficulty |
|----------|-----------|
| `algs/SAPO/sapo.py` | Easy |
| `algs/GSPO/gspo.py` | Easy-Moderate |
| `algs/VAPO/vapo.py` | Moderate |
| `algs/M2PO/m2po.py` | Moderate |
| `algs/RM/rm.py` + `main_rm.py` | Moderate |
| `algs/Distill/distill.py` | Moderate |
| `algs/SPPO/sppo.py` | High |

This gives **6-7 new algorithm files** maximum. But even in this case, most of them would be <100 lines since they inherit from COMMON.

### Where the real work is

The bulk of implementation effort is NOT in new algorithm files. It is in:

1. **Advantage computation refactoring** (~100 lines): Generalize `vllm_engine.py:normalize_rewards()` and add a batch-level normalization pass in `main_rl.py` to support the various advantage modes (z-score, RLOO, mean-only, batch-whitening, token-level returns).

2. **Dynamic group filtering** (~30 lines in `main_rl.py`): Filter prompt groups with uniform rewards before training. Needed for DAPO.

3. **Greedy decode path** (~30 lines in `vllm_engine.py`): Add a greedy generation pass for ReMax baseline.

4. **Reward model training pipeline** (~150 lines): New entry point + BT loss trainer.

5. **Multi-reward interface** (~50 lines): Extend reward functions to return named components for GDPO.

---

## 5. Recommended Implementation Order

### Phase 1: Advantage Refactoring (unlocks 8 algorithms at once)

Refactor the advantage computation to support multiple modes. This single change unlocks Dr.GRPO, RLOO, REINFORCE++ baseline, LitePPO, REINFORCE++, and the advantage component of DAPO.

**Implementation:**

Add `rollout.advantage_mode` config with values:
| Mode | Advantage Formula | Algorithms Using It |
|------|-------------------|-------------------|
| `zscore` (default) | `(r - mean_g) / (std_g + eps)` | GRPO |
| `mean_only` | `r - mean_g` | Dr.GRPO, LitePPO (+ batch std), REINFORCE++ baseline (+ batch whiten) |
| `rloo` | `r - mean_others` = `G/(G-1) * (r - mean_g)` | RLOO |
| `token_returns` | `cumsum(gamma * r)` | REINFORCE++ |
| `greedy_baseline` | `r - r_greedy` | ReMax |

Add optional batch-level post-processing:
| Post-Processing | What It Does | Algorithms Using It |
|-----------------|-------------|-------------------|
| `none` | No batch normalization | Dr.GRPO, RLOO |
| `whiten` | `(A - mean_batch) / std_batch` | REINFORCE++, REINFORCE++ baseline |
| `batch_std` | `A / std_batch` | LitePPO |

**Effort:** ~80 lines total. **Unlocks:** Dr.GRPO, RLOO, REINFORCE++, REINFORCE++ baseline, LitePPO.

### Phase 2: Loss Variants + Dynamic Filtering

| Item | Effort | Unlocks |
|------|--------|---------|
| Dynamic group filtering in `main_rl.py` | ~15 lines | DAPO |
| SAPO sigmoid gate loss (`algs/SAPO/sapo.py`) | ~30 lines | SAPO |
| Dr.GRPO constant divisor loss normalization | ~10 lines | Dr.GRPO (complete) |
| Adaptive KL controller | ~30 lines | Adaptive KL |
| Multiple KL modes in `compute_kl_distance()` | ~40 lines | All KL modes |
| PF-PPO replay resampling | ~25 lines | PF-PPO |
| Multi-iteration loop in `main_rl.py` | ~20 lines | Multi-iter PPO |

**Effort:** ~170 lines total. **Unlocks:** DAPO, SAPO, PF-PPO, Multi-iter PPO, Adaptive KL, KL modes.

### Phase 3: Sequence-Level and Value Extensions

| Item | Effort | Unlocks |
|------|--------|---------|
| GSPO sequence-level ratio (`algs/GSPO/gspo.py`) | ~40 lines | GSPO |
| VAPO length-adaptive GAE + value pretraining | ~70 lines | VAPO |
| M2PO second-moment masking | ~40 lines | M2PO |
| Multi-reward normalization (GDPO) | ~50 lines | GDPO |
| Greedy decode path for ReMax | ~40 lines | ReMax |

**Effort:** ~240 lines total.

### Phase 4: New Training Paradigms

| Item | Effort | Unlocks |
|------|--------|---------|
| Reward model training pipeline | ~150 lines | RM training |
| On-policy distillation | ~60 lines | Distillation |
| SPPO self-play | ~300+ lines | SPPO |

**Effort:** ~500+ lines total.

---

## Summary Table: All 26 Algorithms

| # | Algorithm | Category | Difficulty | LoC | Special Case Of | Needs New File? |
|---|-----------|----------|-----------|-----|-----------------|-----------------|
| 1 | SFT | Already done | -- | 0 | -- | -- |
| 2 | DPO | Already done | -- | 0 | -- | -- |
| 3 | PPO | Already done | -- | 0 | -- | -- |
| 4 | GRPO | Already done | -- | 0 | -- | -- |
| 5 | CISPO | Already done | -- | 0 | -- | -- |
| 6 | P3O | Already done | -- | 0 | -- | -- |
| 7 | Dr.GRPO | Config variant | Trivial | ~10 | GRPO - std + constant denom | No |
| 8 | REINFORCE++ baseline | Config variant | Trivial | ~20 | GRPO mean-only + batch whiten | No |
| 9 | LitePPO | Config variant | Trivial | ~5 | = REINFORCE++ baseline variant | No |
| 10 | DAPO | Config + filter | Easy | ~15 | GRPO + filtering + config | No |
| 11 | RLOO | Advantage change | Trivial | ~10 | GRPO with LOO baseline | No |
| 12 | REINFORCE++ | Advantage change | Easy | ~20 | GRPO with token returns | No |
| 13 | SAPO | New loss | Easy | ~30 | Sigmoid gate replaces clip | Yes |
| 14 | GSPO | New loss | Easy-Mod | ~25 | Seq-level ratio replaces token | Yes |
| 15 | ReMax | Advantage change | Easy-Mod | ~40 | GRPO with greedy baseline | No |
| 16 | VAPO | PPO extension | Moderate | ~70 | PPO + adaptive GAE + pretrain | No (extend PPO) |
| 17 | Multi-iter (mu>1) | Loop change | Easy-Mod | ~20 | Training loop modification | No |
| 18 | PF-PPO | Data preprocessing | Easy | ~25 | Replay buffer resampling | No |
| 19 | M2PO | New mask | Moderate | ~40 | Second-moment token mask | Optional |
| 20 | On-policy distill | New paradigm | Moderate | ~60 | Teacher RKL + RL joint loss | Optional |
| 21 | RM training | New pipeline | Moderate | ~150 | New entry point | Yes |
| 22 | SPPO | New paradigm | High | ~300+ | Iterative self-play | Yes |
| 23 | Adaptive KL | Config extension | Trivial | ~30 | KL controller wrapper | No |
| 24 | KL modes | Config extension | Trivial | ~40 | Extend compute_kl_distance | No |
| 25 | GDPO | Advantage change | Easy-Mod | ~50 | Multi-reward normalization | No |
| 26 | PRIME | Not an algorithm | N/A | N/A | Reward function | No |

**Bottom line:** Of 20 missing items (excluding PRIME which is not an algorithm), **15 are config-level variants or small extensions of existing code** (< 50 lines each). Only **2 need new algorithm files** (SAPO, GSPO). **2 are moderate extensions** (VAPO extends PPO, M2PO adds masking). **1 needs a new pipeline** (RM training). **1 is a research project** (SPPO). The advantage refactoring in Phase 1 (~80 lines) unlocks 5 algorithms simultaneously.
