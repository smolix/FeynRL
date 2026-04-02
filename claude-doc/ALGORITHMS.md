# FeynRL Algorithms -- In-Depth Reference

This document provides a detailed mathematical and implementation-level description of every algorithm in FeynRL. Each section covers the theoretical foundation, the exact loss formulation as implemented, hyperparameters, and references to the original literature.

---

## Table of Contents

1. [Common Infrastructure (COMMON)](#1-common-infrastructure)
2. [Supervised Fine-Tuning (SFT)](#2-supervised-fine-tuning-sft)
3. [Direct Preference Optimization (DPO)](#3-direct-preference-optimization-dpo)
4. [Group Relative Policy Optimization (GRPO)](#4-group-relative-policy-optimization-grpo)
5. [Proximal Policy Optimization (PPO)](#5-proximal-policy-optimization-ppo)
6. [Conservative In-Sample Policy Optimization (CISPO)](#6-conservative-in-sample-policy-optimization-cispo)
7. [P3O: ESS-Based Policy Optimization](#7-p3o-ess-based-policy-optimization)
8. [Loss Normalization](#8-loss-normalization)

---

## 1. Common Infrastructure

**File:** `algs/RL/common.py`

The `COMMON` base class provides infrastructure shared by all RL algorithms (PPO, GRPO, CISPO, P3O). It is not used by SFT or DPO.

### 1.1 Policy Forward Pass

Given input tokens `x = [x_1, ..., x_T]`, the policy model produces logits at each position. Log-probabilities are computed via cross-entropy in float32 for numerical stability:

```
logprobs[t] = log pi_theta(x_{t+1} | x_{1:t})     for t in [1, T-1]
```

The output is prediction-aligned: `logprobs[t]` is the log-probability assigned to token `x_{t+1}` by the logit at position `t`. Shape: `[B, T-1]`.

**Entropy** (optional, computed only when `entropy_coeff > 0`):
```
H[t] = -sum_v pi_theta(v | x_{1:t}) * log pi_theta(v | x_{1:t})
```

Computed via `torch.distributions.Categorical(logits=...).entropy()`.

### 1.2 Reference Forward Pass

Identical to the policy forward pass but using the frozen reference model `pi_ref` under `torch.no_grad()`. Used for KL penalty computation.

### 1.3 KL Divergence

FeynRL uses the **variance-reduced KL estimator** (also called the "Schulman KL" or "unbiased KL"):

```
KL[t] = log(pi_theta / pi_ref) + (pi_ref / pi_theta) - 1
      = log_ratio + exp(-log_ratio) - 1
```

where `log_ratio = logprobs[t] - ref_logprobs[t]`.

This estimator is always non-negative and has lower variance than the naive `log(pi/pi_ref)` estimator. It equals zero when `pi_theta = pi_ref`.

**Reference:** Schulman, J. (2020). "Approximating KL Divergence." http://joschu.net/blog/kl-approx.html

### 1.4 Global Token Normalization

When `normalize_loss=True`, the loss is normalized by the total number of valid tokens across all ranks and all micro-batches in the gradient accumulation window:

```
ga_denom = all_reduce_sum(local_token_count)
dp_scale = ga_steps * world_size
scaled_loss = loss_sum * (dp_scale / ga_denom)
```

The `dp_scale` factor cancels DeepSpeed's internal averaging (which divides by `ga_steps` and by `world_size`), replacing it with true global per-token normalization. This ensures each token contributes equally to the gradient regardless of sequence length variance across ranks.

### 1.5 Logprob Sanitization

Before loss computation, logprobs are sanitized to prevent NaN propagation:

```python
logprobs = torch.nan_to_num(logprobs, nan=0.0, posinf=0.0, neginf=0.0)
```

A warning is logged when sanitization fires.

---

## 2. Supervised Fine-Tuning (SFT)

**File:** `algs/SFT/sft.py`

### 2.1 Overview

SFT trains the model to predict the next token in supervised (prompt, response) pairs. The loss is standard cross-entropy, masked to exclude prompt tokens and padding.

### 2.2 Loss Function

Given a sequence `x = [x_1, ..., x_T]` with a loss mask `m` that is 1 on response tokens and 0 on prompt/padding tokens:

```
L_SFT = -sum_{t=1}^{T-1} m[t] * log p_theta(x_{t+1} | x_{1:t})
```

The raw loss sum `L_SFT` is a scalar. The final loss for backward depends on the normalization mode:

**With `normalize_loss=True` (global token normalization):**
```
loss = L_SFT * (ga_steps * world_size) / ga_denom
```
where `ga_denom` is the total valid token count across all GPUs in the gradient accumulation window.

**With `normalize_loss=False` (scaled sum):**
```
loss = L_SFT * ga_steps * world_size
```

### 2.3 Interface

```python
class SFT:
    def train_step(self, micro_batch, ga_denom=None, ga_steps=1) -> dict
    def eval_step(self, micro_batch) -> dict  # returns {loss_sum, num_tokens}
    def compute_loss(self, logits, target_ids, loss_mask, ga_denom, ga_steps, is_training) -> (loss, loss_sum, num_tokens)
```

### 2.4 Data Format

- `input_ids`: `[B, T]` -- concatenation of prompt + response + padding
- `attn_mask`: `[B, T]` -- 1 for real tokens, 0 for padding
- `loss_mask`: `[B, T-1]` -- 1 for response tokens (training targets), 0 for prompt/padding

The loss mask is prediction-aligned: `loss_mask[t]` indicates whether to compute loss for predicting token `x_{t+1}`.

### 2.5 Reference

- Standard language model fine-tuning. See: Radford et al. (2018), "Improving Language Understanding by Generative Pre-Training."

---

## 3. Direct Preference Optimization (DPO)

**File:** `algs/DPO/dpo.py`

### 3.1 Overview

DPO learns from paired preference data (chosen vs. rejected responses) without requiring a separate reward model or RL training loop. It directly optimizes the policy to assign higher implicit reward to chosen responses.

### 3.2 Loss Function

Given a prompt `x`, chosen response `y_w`, and rejected response `y_l`:

**Step 1: Length-normalized log-ratios**

```
r_w = (1/|y_w|) * sum_{t} [log pi_theta(y_w^t | x, y_w^{<t}) - log pi_ref(y_w^t | x, y_w^{<t})]
r_l = (1/|y_l|) * sum_{t} [log pi_theta(y_l^t | x, y_l^{<t}) - log pi_ref(y_l^t | x, y_l^{<t})]
```

Length normalization (`/|y|`) prevents the model from favoring shorter responses.

**Step 2: DPO loss**

```
L_DPO = -log sigma(beta * (r_w - r_l))
```

where `sigma` is the sigmoid function and `beta` is the inverse temperature parameter.

The final loss is averaged over the batch: `loss = mean(L_DPO)`.

### 3.3 Implementation Details

- Chosen and rejected sequences are stacked as `[B, 2, T]` and reshaped to `[2B, T]` for a single forward pass
- Even rows (0::2) are chosen, odd rows (1::2) are rejected
- Reference logprobs are computed in a single `torch.no_grad()` forward pass
- The reference model is set to `eval()` mode from initialization and never updated
- Cross-entropy is computed in float32 for numerical stability

### 3.4 Hyperparameters

| Parameter | Config Key | Description |
|-----------|-----------|-------------|
| `beta` | `train.cl_beta` | Inverse temperature. Higher values make the model more sensitive to preference differences. Typical range: 0.1--0.5 |

### 3.5 Metrics

| Metric | Description |
|--------|-------------|
| `loss` | DPO loss value |
| `chosen_rewards` | Mean implicit reward for chosen responses |
| `rejected_rewards` | Mean implicit reward for rejected responses |
| `reward_accuracies` | Fraction where chosen reward > rejected reward |

### 3.6 Reference

- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." *NeurIPS 2023*. https://arxiv.org/abs/2305.18290

---

## 4. Group Relative Policy Optimization (GRPO)

**File:** `algs/GRPO/grpo.py`

### 4.1 Overview

GRPO is a simplified policy gradient method that eliminates the need for a learned value function. Instead of using GAE for advantage estimation, it uses z-score normalized rewards computed from multiple completions per prompt as advantages. This makes it significantly simpler than PPO while maintaining competitive performance.

### 4.2 Advantage Estimation

For each prompt, `n_samples` completions are generated. The scalar reward for each completion is z-score normalized within the group:

```
A_i = (R_i - mu_group) / (sigma_group + eps)
```

where `mu_group` and `sigma_group` are the mean and standard deviation of rewards for that prompt's completions. This z-score is broadcast across all tokens in the response (or optionally only placed on the last token).

### 4.3 Loss Function

The policy loss uses PPO-style clipped importance sampling:

```
ratio[t] = pi_theta(a_t | s_t) / pi_old(a_t | s_t) = exp(logprobs[t] - old_logprobs[t])
L_clip[t] = -min(ratio[t] * A[t], clip(ratio[t], 1-c_low, 1+c_high) * A[t])
```

**Full loss per micro-batch:**

```
L_total = sum_t [L_clip[t] * mask[t]]           # policy loss (sum over valid tokens)
        - ent_coeff * sum_t [H[t] * mask[t]]     # entropy bonus (encourages exploration)
        + kl_coeff  * sum_t [KL[t] * mask[t]]    # KL penalty (stay close to reference)
```

All terms are raw sums; normalization is applied before backward based on the normalization mode (see Section 8).

### 4.4 Implementation Details

- Micro-batches are shuffled with a deterministic RNG before processing to avoid systematic bias from gradient accumulation boundary placement
- The `update_after_full_replay` flag controls whether the optimizer step happens after each GA window (standard) or after the entire replay buffer shard (treating it as one big batch)
- Advantages come from the `zscore` field in the replay buffer, pre-computed during rollout collection

### 4.5 Hyperparameters

| Parameter | Config Key | Description |
|-----------|-----------|-------------|
| `clip_low` | `train.clip_low` | Lower clipping bound for ratio. Typical: 0.2 |
| `clip_high` | `train.clip_high` | Upper clipping bound for ratio. Typical: 0.2 (symmetric) or different (asymmetric) |
| `kl_coeff` | `train.kl_coeff` | KL penalty coefficient. 0.0 = no KL penalty |
| `entropy_coeff` | `train.entropy_coeff` | Entropy bonus coefficient. 0.0 = no entropy bonus |
| `n_samples` | `rollout.n_samples` | Completions per prompt for z-score computation |
| `eps_reward_norm` | `reward.eps_reward_norm` | Epsilon for z-score denominator stability |

### 4.6 Metrics

| Metric | Description |
|--------|-------------|
| `pi_loss` | Per-token mean policy loss |
| `clipfrac` | Fraction of tokens where ratio was clipped |
| `approx_kl` | Approximate KL divergence from old policy (variance-reduced) |
| `kl_ref` | KL divergence from reference policy |
| `ent_loss` | Per-token mean entropy |
| `loss_total` | Total per-token mean loss |

### 4.7 Reference

- Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Zhang, M., ... & Guo, D. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." https://arxiv.org/abs/2402.03300

---

## 5. Proximal Policy Optimization (PPO)

**File:** `algs/PPO/ppo.py`, `algs/PPO/value_net.py`

### 5.1 Overview

PPO is the canonical actor-critic RL algorithm for language model post-training. It uses a learned value function for advantage estimation via Generalized Advantage Estimation (GAE), and clips importance ratios to constrain policy updates. FeynRL's PPO implementation trains policy and value models in parallel with separate DeepSpeed engines.

### 5.2 Value Network

**File:** `algs/PPO/value_net.py`

The `ValueNetwork` wraps a causal LM backbone, replacing the language model head (`hidden_dim -> vocab_size`) with a scalar value head (`hidden_dim -> 1`):

```python
class ValueNetwork(nn.Module):
    backbone: transformer backbone (from base LM, LM head removed)
    value_head: nn.Linear(hidden_size, 1, bias=False)  # initialized to zeros
```

Output shape: `[B, T, 1]`, squeezed to `[B, T]`.

The value head is zero-initialized so initial value predictions don't dominate early training. The backbone can be shared architecture with the policy but loaded from a separate checkpoint.

### 5.3 Generalized Advantage Estimation (GAE)

**Method:** `PPO.compute_advantages(rewards, values, done, mask, last_val)`

GAE computes temporal-difference advantages with exponential weighting:

```
delta[t] = r[t] + gamma * V[t+1] * (1 - done[t]) - V[t]
A[t] = delta[t] + gamma * tau * A[t+1] * (1 - done[t])
```

where:
- `gamma` is the discount factor
- `tau` (lambda) controls the bias-variance tradeoff (higher = lower bias, higher variance)
- `done[t] = 1` at terminal tokens (EOS/stop), which cuts off bootstrapping
- `V[t]` comes from the frozen value network (computed before any updates)

**Returns** are computed as `R[t] = A[t] + V[t]`.

**Key implementation details:**
- Computation is done in float32 for numerical stability
- Padding positions are masked: `rewards` and `values` are filled with 0 at invalid positions
- The mask must be contiguous (no holes) -- validated with a drop/rise transition check
- `last_val` provides bootstrap value for the last position (used when the sequence was truncated, not terminated)

**Global advantage normalization:** After computing GAE for all micro-batches, advantages are z-score normalized across all ranks:

```python
global_mean = all_reduce_sum(local_sum) / all_reduce_sum(local_count)
global_std  = sqrt(all_reduce_sum(local_sq_sum) / global_count) + 1e-8
normalized_adv = (adv - global_mean) / global_std
```

This uses two all-reduce operations (for mean and variance) with float64 precision for numerical stability.

### 5.4 Policy Loss

Identical to GRPO (Section 4.3), using clipped importance ratios. The only difference is the source of advantages: PPO uses GAE-computed advantages instead of z-score rewards.

### 5.5 Value Loss

```
L_value[t] = 0.5 * (V_theta(s_t) - R[t])^2
L_value = sum_t [L_value[t] * mask[t]]
```

where `R[t]` are the GAE returns (detached). The 0.5 factor is a convention for cleaner gradients.

### 5.6 Training Loop

PPO's `train_step` differs from GRPO in two key ways:

1. **Pre-computation phase:** Before any gradient updates, GAE is computed for all micro-batches using the frozen value network. This ensures consistent advantage estimates across the entire training step.

2. **Dual engine updates:** Policy and value engines are updated in alternation within each micro-batch:
   ```
   for each micro-batch:
       policy_forward -> policy_loss -> policy_backward -> policy_step
       value_forward  -> value_loss  -> value_backward  -> value_step
   ```

Both engines share the same gradient accumulation schedule.

### 5.7 Hyperparameters

All GRPO hyperparameters plus:

| Parameter | Config Key | Description |
|-----------|-----------|-------------|
| `tau` | `train.tau` | GAE lambda. 1.0 = Monte Carlo returns, 0.0 = 1-step TD. Typical: 0.95 |
| `gamma` | `train.gamma` | Discount factor. Typical: 1.0 (undiscounted) or 0.99 |
| `value_lr` | `train.value_lr` | Learning rate for value network (defaults to `train.lr`) |
| `value_model` | `model.value_model` | Path to value model checkpoint |

### 5.8 Metrics

All GRPO metrics plus:

| Metric | Description |
|--------|-------------|
| `value_loss_v` | Per-token mean value loss |

### 5.9 References

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). "Proximal Policy Optimization Algorithms." https://arxiv.org/abs/1707.06347
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation." https://arxiv.org/abs/1506.02438
- Ouyang, L., Wu, J., Jiang, X., et al. (2022). "Training language models to follow instructions with human feedback." *NeurIPS 2022*. https://arxiv.org/abs/2203.02155

---

## 6. Conservative In-Sample Policy Optimization (CISPO)

**File:** `algs/CISPO/cispo.py`

### 6.1 Overview

CISPO is a conservative variant of policy gradient that uses the importance ratio as a detached weighting coefficient rather than as part of the gradient. This makes the policy update more conservative by reducing the influence of large likelihood ratio changes: the gradient flows only through `log pi_theta`, not through the ratio itself.

### 6.2 Loss Function

```
ratio[t] = exp(logprobs[t] - old_logprobs[t])
rho[t] = clip(ratio[t], 1 - c_low, 1 + c_high)
L_CISPO = -sum_t [rho[t].detach() * log pi_theta(a_t | s_t) * A[t] * mask[t]]
```

The critical difference from PPO/GRPO: `rho[t]` is **detached** from the computation graph. The gradient only flows through `log pi_theta`:

```
grad L_CISPO / grad theta = -sum_t [rho[t].detach() * A[t] * grad log pi_theta / grad theta]
```

This is equivalent to a weighted policy gradient where the weights are bounded importance ratios. When `ratio ~ 1` (policy hasn't changed much), this is equivalent to standard REINFORCE with advantage weighting.

### 6.3 Entropy and KL Terms

Same as GRPO:
```
L_total = L_CISPO - ent_coeff * sum_t [H[t] * mask[t]] + kl_coeff * sum_t [KL[t] * mask[t]]
```

### 6.4 Advantages

Same as GRPO: z-score normalized rewards from the replay buffer. CISPO does not use a value function.

### 6.5 Hyperparameters

Same as GRPO (Section 4.5).

### 6.6 Comparison with GRPO

| Aspect | GRPO | CISPO |
|--------|------|-------|
| Gradient path | Through `min(ratio*A, clip(ratio)*A)` | Through `log pi` only, ratio is detached |
| Update magnitude | Can be aggressive when ratio is large | More conservative: large ratios are bounded, and no gradient through ratio |
| Clipping mechanism | min-clipping (PPO-style) | Multiplicative weight clipping |

### 6.7 Reference

- Related to RLOO / In-sample policy gradient approaches. The specific CISPO formulation follows conservative importance weighting principles.

---

## 7. P3O: ESS-Based Policy Optimization

**File:** `algs/P3O/p3o.py`

### 7.1 Overview

P3O adapts the clipping bound dynamically based on the **Effective Sample Size (ESS)**, a measure of how far the current policy has drifted from the rollout policy. When the policies are similar (ESS near 1), P3O allows larger updates. When they diverge (ESS near 0), it becomes more conservative.

### 7.2 Effective Sample Size

ESS measures the quality of importance sampling:

```
ESS = (sum_t w[t])^2 / (sum_t w[t]^2) / n
```

where `w[t] = ratio[t] = pi_theta / pi_old` and `n` is the number of valid (non-padded) tokens. ESS is in `(0, 1]`:
- ESS = 1: policies are identical (all weights equal)
- ESS -> 0: policies have diverged significantly (weights are concentrated)

Only valid (non-padded) positions are used in the ESS calculation. Padded positions have `ratio = exp(0) = 1.0` which would bias ESS toward 1.0.

### 7.3 Loss Function

```
ratio[t] = exp(logprobs[t] - old_logprobs[t])
ess = ESS(ratio, mask)
rho[t] = clip(ratio[t], 0, ess)
L_P3O = -sum_t [rho[t].detach() * log pi_theta(a_t | s_t) * A[t] * mask[t]]
```

Like CISPO, `rho[t]` is detached. The key difference: the upper clipping bound is ESS itself, not a fixed hyperparameter. This provides **adaptive trust region** control:
- When ESS is high (policies similar), the effective clipping range `[0, ess]` is wider, allowing larger updates
- When ESS is low (policies diverged), the range narrows, constraining the update

### 7.4 Entropy and KL Terms

Same as GRPO/CISPO.

### 7.5 Advantages

Same as GRPO/CISPO: z-score normalized rewards.

### 7.6 Hyperparameters

Same as GRPO, plus the ESS metric is logged. Note that `clip_low` and `clip_high` are accepted for interface compatibility but are **not used** in the P3O loss calculation -- only for the `clipfrac` diagnostic metric.

### 7.7 Metrics

All GRPO metrics plus:

| Metric | Description |
|--------|-------------|
| `ess_factor` | Current ESS value. Indicates policy divergence from rollout policy |

### 7.8 Reference

- Related to Effective Sample Size concepts in importance sampling. See: Kong, A. (1992). "A note on importance sampling using standardized weights." Technical report, University of Chicago.
- Adaptive trust region methods for policy optimization.

---

## 8. Loss Normalization

All algorithms support two normalization modes, controlled by `train.normalize_loss`:

### 8.1 Global Token Normalization (`normalize_loss=True`)

Each token contributes equally to the gradient, regardless of how sequences are distributed across ranks:

```
ga_denom = all_reduce_sum(local_valid_tokens)  # total tokens across all ranks, all micro-batches
dp_scale = ga_steps * world_size               # cancel DeepSpeed's internal averaging
loss_for_backward = loss_sum * (dp_scale / ga_denom)
```

This is the recommended mode for distributed training with variable-length sequences.

### 8.2 Per-Micro-Batch Normalization (`normalize_loss=False`)

Each micro-batch is normalized independently by its local token count:

```
loss_for_backward = loss_sum / local_denom
```

With a correction factor for incomplete gradient accumulation buckets:
- If `num_micro_batches % ga_steps != 0`, the last bucket has fewer micro-batches
- DeepSpeed still divides by `ga_steps`, so the loss is scaled up by `ga_steps / remainder`

With `update_after_full_replay=True`, the entire replay shard is treated as one big GA bucket, and the correction becomes `ga_steps / num_micro`.

---

## Algorithm Comparison Summary

| Algorithm | Value Function | Advantage Source | Clipping Style | Gradient Through Ratio | Models Required |
|-----------|---------------|-----------------|---------------|----------------------|-----------------|
| **SFT** | -- | -- | -- | -- | 1 (policy) |
| **DPO** | -- | Implicit (preference) | -- | -- | 2 (policy + frozen ref) |
| **GRPO** | No | Z-score rewards | PPO-style min-clip | Yes | 1-2 (policy + optional ref) |
| **PPO** | Yes (learned) | GAE | PPO-style min-clip | Yes | 2-3 (policy + value + optional ref) |
| **CISPO** | No | Z-score rewards | Detached clipped weight | No (detached) | 1-2 (policy + optional ref) |
| **P3O** | No | Z-score rewards | ESS-adaptive detached | No (detached) | 1-2 (policy + optional ref) |
