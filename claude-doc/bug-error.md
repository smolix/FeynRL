# Code Review: Bugs and Errors

Systematic code review of all changes on the `feature/missing-algorithms` branch. Findings are classified by severity.

---

## Critical Bugs (will cause runtime failures)

### BUG-1: New algorithm parameters not passed through `create_training_engines`

**Files:** `main_rl.py:40-105`, `algs/SAPO/sapo.py`, `algs/M2PO/m2po.py`, `algs/PPO/ppo.py`

**Problem:** `create_training_engines()` builds the `kwargs` dict that is passed to every algorithm's `__init__`. It has special-case logic for PPO (lines 72-78) but **no special-case logic** for SAPO, GSPO, M2PO, or PPO's new VAPO parameters. When these algorithms are instantiated, the required parameters will be missing:

- **SAPO** requires `sapo_tau_pos` and `sapo_tau_neg` -- these have defaults in the `__init__` signature, so SAPO will silently use defaults (1.0, 1.05) **even if the user configures different values in YAML**. This is a silent config-ignored bug.
- **M2PO** requires `m2_threshold` -- same issue, silently uses default 0.2.
- **PPO + VAPO** requires `vapo_enabled`, `vapo_alpha`, `vapo_nll_weight` -- same issue, VAPO extensions silently disabled.

**Fix needed:** Add parameter forwarding in `create_training_engines()`:
```python
if alg_name == 'sapo':
    kwargs['sapo_tau_pos'] = params.train.sapo_tau_pos or 1.0
    kwargs['sapo_tau_neg'] = params.train.sapo_tau_neg or 1.05
if alg_name == 'm2po':
    kwargs['m2_threshold'] = params.train.m2_threshold or 0.2
if alg_name == 'ppo':
    # ... existing PPO kwargs ...
    kwargs['vapo_enabled'] = params.train.vapo_enabled
    kwargs['vapo_alpha'] = params.train.vapo_alpha
    kwargs['vapo_nll_weight'] = params.train.vapo_nll_weight
```

### BUG-2: `advantage_mode` not passed to rollout engine

**Files:** `main_rl.py:107-177`, `rollouts/vllm_engine.py:42`

**Problem:** `create_rollout_engines()` builds `kwargs` for `VLLMRolloutEngine.__init__()` but **does not include `advantage_mode`**. The new parameter was added to the engine's `__init__` with a default of `"zscore"`, but even if the user sets `train.advantage_mode` in config, it will never reach the rollout engine. All advantage modes other than zscore will silently not work.

**Fix needed:** Add to `create_rollout_engines()`:
```python
kwargs['advantage_mode'] = params.train.advantage_mode or "zscore"
```

### BUG-3: `_kl_mode` attribute never set on algorithm instances

**Files:** `algs/RL/common.py:172-173`

**Problem:** `compute_kl_distance()` reads `self._kl_mode` via `getattr(self, '_kl_mode', 'k3')` but **no algorithm's `__init__` ever sets `self._kl_mode`**. The config field `train.kl_mode` exists but is never wired through `create_training_engines()` to the algorithm instance. The KL mode will always fall back to `'k3'` regardless of config.

**Fix needed:** Either:
1. Add `kl_mode` to the kwargs in `create_training_engines()` and store it as `self._kl_mode` in each algorithm's `__init__`, or
2. Pass `kl_mode` through the COMMON base class `__init__` (would require changing all algorithm constructors).

### BUG-4: `AdaptiveKLController` never instantiated or integrated

**Files:** `algs/RL/common.py:18-37`, `main_rl.py`

**Problem:** The `AdaptiveKLController` class exists but is **never instantiated anywhere**. The config fields `train.kl_control`, `train.kl_target`, and `train.kl_horizon` exist but nothing reads them. The controller's `update()` method is never called. The adaptive KL feature is entirely dead code.

**Fix needed:** The controller should be instantiated in the algorithm's `__init__` when `kl_control == "adaptive"`, and `update()` should be called after each training step with the observed `approx_kl` metric. The updated `kl_coeff` should be used in subsequent loss computations.

### BUG-5: `loss_denom_mode: "constant"` not implemented

**Files:** `configs/load.py:107`, all algorithm `compute_policy_loss` methods

**Problem:** The config field `train.loss_denom_mode` accepts `"constant"` for Dr.GRPO-style loss normalization (divide by `B * max_seq_len` instead of actual token count), but **no algorithm reads this field or implements the constant denominator logic**. Dr.GRPO's second key modification (constant divisor) is documented but not functional.

**Fix needed:** In each algorithm's `train_step()`, when `loss_denom_mode == "constant"`, replace `local_denom` (actual token count) with `B * max_seq_len`.

---

## Moderate Bugs (incorrect behavior in specific scenarios)

### BUG-6: SAPO gate function does not preserve gradient through `ratio`

**File:** `algs/SAPO/sapo.py:138-140`

**Problem:** The SAPO gate function is:
```python
taus = torch.where(adv > 0, self.tau_pos, self.tau_neg)
gates = torch.sigmoid(taus * (ratio - 1.0)) * (4.0 / taus)
pi_sum = -(gates * adv * mask).sum()
```

The gradient flows through `gates` which depends on `ratio`, which depends on `logprobs`. This is correct mathematically. However, the `taus` tensor is created from `adv` (which is detached), so `torch.where` creates a new tensor that is not connected to any gradient -- this is fine. But there is a subtle issue: `self.tau_pos` and `self.tau_neg` are Python floats. When used in `torch.where` with a tensor condition, the result is a tensor, but the multiplication `taus * (ratio - 1.0)` works correctly. **This is actually NOT a bug** upon closer inspection -- the gradient path through `sigmoid(taus * (ratio - 1.0))` to `ratio` to `logprobs` is intact.

**Status:** False alarm. Downgraded from bug to note.

### BUG-7: VAPO NLL loss computed after policy loss scaling, causing incorrect gradient magnitude

**File:** `algs/PPO/ppo.py:593-601`

**Problem:** The VAPO NLL loss is added to `pi_loss` **after** the policy loss has already been scaled for backward:
```python
# pi_loss already scaled by dp_scale/ga_denom or local_denom
pi_loss = loss_total_sum * (dp_scale / ga_denom)
# ...
# NLL loss added with raw normalization (/ nll_token_count), NOT scaled by dp_scale/ga_denom
nll_loss = -(pi_logprobs * nll_mask).sum() / nll_token_count
pi_loss = pi_loss + self.vapo_nll_weight * nll_loss
```

The policy loss has been scaled by DeepSpeed's GA/DP scaling factor, but the NLL loss is normalized independently. This means the **relative weight** of NLL vs. policy loss changes depending on `ga_steps` and `world_size`. The NLL loss should be scaled by the same factor as the policy loss, or the addition should happen before scaling.

**Fix needed:** Either compute `nll_loss` with the same scaling as `pi_loss`, or add the NLL auxiliary term to `loss_total_sum` before scaling in the `if self.normalize_loss:` block.

### BUG-8: `filter_uniform_groups` applied per-batch, not across the full rollout

**File:** `main_rl.py:516-519` (in `collect_rollouts`)

**Problem:** Group filtering is applied inside the per-batch loop:
```python
for rollout_batch in dataloader:
    # ... generate ...
    if filter_groups:
        rollout_merged = filter_uniform_groups(rollout_merged, n_samples, logger=logger)
    replay_buffer.add_batch_seqs(rollout_merged)
```

`rollout_merged` here is the merge of a **single batch** of rollouts. If prompts from the same group are split across multiple batches (which shouldn't normally happen since each batch is processed independently), this works correctly. However, the function groups by `prompt_ids`, and within a single batch, all N samples for a prompt should be present. **This is actually correct for the normal case.**

But there's a subtle issue: `rollout_merged` was already passed through `merge_rollout_with_stats()` which collected stats. The filtering happens **after** stats collection, so the reported rollout statistics (reward mean, pass@k, etc.) include the filtered-out groups. This means logged metrics don't match what's actually trained on.

**Fix needed:** Either filter before `merge_rollout_with_stats()`, or recompute stats after filtering.

### BUG-9: `apply_batch_advantage_norm` reads wrong key from replay buffer items

**File:** `main_rl.py:373-380`

**Problem:**
```python
for item in replay_buffer.items:
    mask = item["masks"]
    zscores = item["zscores"]
```

The replay buffer stores items with keys from `ReplayBuffer.add()` (line 120): `"masks"` and `"zscores"`. This matches. But the collated output from `collate_fn` uses different keys: `"mask"` (no 's') and `"zscore"` (no 's'). The function operates on `replay_buffer.items` (pre-collation), so it uses the correct keys (`"masks"`, `"zscores"`). **This is correct.**

However, the in-place modification `item["zscores"] = (item["zscores"] - batch_mean) / batch_std` modifies the raw items, and the `collate_fn` later reads from these same items. Since `collate_fn` reads `x["zscores"]`, the modification propagates correctly. **This is also correct.**

**Status:** Not a bug after thorough analysis.

### BUG-10: GDPO grouping by first 20 tokens is fragile

**File:** `main_rl.py:429-430`

**Problem:**
```python
key = tuple(item["input_ids"][:20].tolist())
```

Using the first 20 tokens as a prompt key is a heuristic. If two different prompts share the same first 20 tokens (possible with similar system prompts), they will be incorrectly grouped together, causing wrong normalization. The original `filter_uniform_groups` function (line 335) uses the full `prompt_ids`, which is correct. The GDPO function should do the same, but the replay buffer doesn't store `prompt_ids` separately -- only `input_ids` (prompt + response concatenated).

**Impact:** Incorrect GDPO normalization when prompts share prefixes. Rare but possible, especially with short prompts or shared system prompts.

**Fix needed:** Use a hash of the full prompt portion of `input_ids` (up to the first response token), or store `prompt_len` in the replay buffer items to extract the exact prompt.

### BUG-11: `compute_distillation_loss` returns raw sum but is never called

**Files:** `algs/RL/common.py:146-160`, all algorithm `train_step` methods

**Problem:** `compute_distillation_loss()` and `teacher_forward()` are implemented in COMMON and the teacher model is loaded in `init_training_engine()`, but **no algorithm's `train_step()` ever calls them**. The distillation feature is dead code at the algorithm level. Users who set `model.teacher_model` and `train.distill_weight` will load the teacher model (consuming GPU memory) but never use it for training.

**Fix needed:** The distillation loss needs to be integrated into each algorithm's `train_step()` (or at least GRPO's, since distillation is most commonly used with GRPO). After computing the policy loss:
```python
if hasattr(self, 'teacher_engine') and self.teacher_engine is not None:
    teacher_logprobs = self.teacher_forward(input_ids, att_mask, pos_ids)
    distill_loss, _ = self.compute_distillation_loss(pi_logprobs, teacher_logprobs, mask)
    loss_total_sum = loss_total_sum + self.distill_weight * distill_loss
```

### BUG-12: `distill_weight` attribute never set on any algorithm

**Files:** `algs/RL/common.py:408`, all algorithm `__init__` methods

**Problem:** In `init_training_engine()`, the teacher model loading checks `distill_weight = getattr(self, 'distill_weight', 0.0)`. But **no algorithm ever sets `self.distill_weight`**, and it's not in `create_training_engines()` kwargs. The teacher model will never be loaded because `distill_weight` always returns `0.0`.

**Fix needed:** Add `distill_weight` to the kwargs pipeline:
```python
# create_training_engines:
kwargs['distill_weight'] = params.train.distill_weight or 0.0
kwargs['teacher_model_path'] = params.model.teacher_model
```
But there's also no `train.distill_weight` config field. It needs to be added to the Train config class.

---

## Low-Severity Issues (edge cases, style, or minor inconsistencies)

### BUG-13: `token_returns` advantage mode places raw reward only on last token

**File:** `rollouts/vllm_engine.py:666-668`

**Problem:** When `advantage_mode == "token_returns"`, the code stores the raw reward on the last token position only:
```python
zscore[-1] = sample['token_rewards'][-1]
```

For REINFORCE++, the advantage should be **token-level discounted cumulative returns**, not a scalar. The current implementation stores the raw scalar reward, and the cumulative return computation is supposed to happen "later in the training loop" per the comment -- but **no training loop code implements this**. The REINFORCE++ algorithm won't have correct token-level returns; it will just use the scalar reward broadcast to all tokens.

**Fix needed:** Implement the discounted cumulative return computation either:
1. In the rollout engine's `token_returns` mode, or
2. In the training loop before passing advantages to the algorithm, or
3. In a custom `compute_policy_loss()` method in a REINFORCE++ algorithm class.

### BUG-14: `num_iterations` config field exists but has no effect

**File:** `configs/load.py:121`

**Problem:** The `train.num_iterations` field is defined but never read by any code. The documentation in GAPS-Algorithms.md states that multi-iteration PPO is "already supported via `train.train_steps_per_epoch`", which is true -- but the `num_iterations` field creates false expectations. Users might set it expecting mu > 1 behavior without realizing they need to use `train_steps_per_epoch` instead.

**Fix:** Either remove the field and document the `train_steps_per_epoch` approach, or implement proper mu > 1 behavior that generates once and trains `num_iterations` times before regenerating.

### BUG-15: Config validation missing for new fields

**Files:** `configs/load.py:524-847`

**Problem:** The `load_and_verify()` function validates many config fields but has **no validation** for any of the new fields:
- `advantage_mode` is not validated against the set of valid values (`zscore`, `mean_only`, `rloo`, `token_returns`, `greedy_baseline`)
- `kl_mode` is not validated against valid values (`k1`, `k2`, `k3`, `abs`, `k3_plus`)
- `kl_control` is not validated against valid values (`fixed`, `adaptive`)
- `sapo_tau_pos` and `sapo_tau_neg` are not validated to be positive
- `m2_threshold` is not validated to be positive
- `loss_denom_mode` is not validated
- `advantage_batch_norm` is not validated
- No check that `alg_name == "sapo"` when SAPO-specific fields are set (and similarly for M2PO, VAPO)
- No check that `kl_target > 0` when `kl_control == "adaptive"`

**Impact:** Users can set invalid values that will cause cryptic runtime errors instead of clean config validation errors.

### BUG-16: `create_training_engines` doesn't handle SAPO/GSPO/M2PO algorithm names for ref model validation

**File:** `configs/load.py:714-718`

**Problem:** The config validation checks:
```python
if config.model.ref_model is not None and (config.train.kl_coeff == 0 or config.train.kl_coeff is None):
    raise ValueError(f"kl_coeff must be > 0 if model.ref is not None")
```

This works for existing algorithms. But SAPO, GSPO, and M2PO can all use a reference model for KL penalty (they inherit this from GRPO). The validation is algorithm-agnostic, so it works correctly. **Not a bug.** (Noting for completeness.)

### BUG-17: M2PO masking loop is O(B * T * T log T) which may be slow

**File:** `algs/M2PO/m2po.py:141-162`

**Problem:** The M2PO second-moment masking uses a Python for-loop over the batch dimension, and within each sequence, sorts the M2 values:
```python
for b in range(m2_values.shape[0]):
    valid_m2 = m2_values[b, valid_idx]
    sorted_m2, sort_idx = valid_m2.sort(descending=True)
    suffix_sums = sorted_m2.flip(0).cumsum(0).flip(0)
    ...
```

For large batch sizes and long sequences, this Python loop + per-sequence sort is slow compared to fully vectorized operations. With `B=32, T=2048`, this does 32 sorts of up to 2048 elements inside the training loop.

**Impact:** Performance degradation with large batches. Not a correctness bug.

**Fix suggestion:** Vectorize using batch-level operations where possible, or move to C++/CUDA kernel.

### BUG-18: `returns_gamma` config field exists but is never used

**File:** `configs/load.py:109`

**Problem:** `train.returns_gamma` is defined for the `token_returns` advantage mode (REINFORCE++), but since the token-level return computation is not implemented (see BUG-13), this field is dead config.

### BUG-19: GDPO `_gdpo_adv` key left in items if exception occurs

**File:** `main_rl.py:436-457`

**Problem:** `apply_gdpo_normalization` adds `"_gdpo_adv"` to each item, then replaces `"zscores"` with it via `item.pop("_gdpo_adv")`. If an exception occurs between adding `"_gdpo_adv"` and the pop (e.g., a KeyError on a missing reward field), the items will have both `"zscores"` (original) and `"_gdpo_adv"` (partial), leaving the buffer in an inconsistent state.

**Impact:** Low. The exception would propagate and crash training anyway. But for robustness, the operation should be atomic or use try/finally.

### BUG-20: `PF-PPO` resampling with replacement creates duplicate tensor references

**File:** `rollouts/replay_buffer.py:232`

**Problem:**
```python
self.items = [self.items[i] for i in indices.tolist()]
```

This creates a new list where multiple entries may reference the **same dict**. If any downstream code modifies items in-place (e.g., GDPO's `item["_gdpo_adv"]` assignment), it will modify all copies. The replay buffer's `collate_fn` reads items but doesn't modify them, so this is safe in the current usage. But it's a latent bug if any future code modifies replay buffer items after PF-PPO resampling.

**Fix suggestion:** Deep-copy resampled items: `[{k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.items[i].items()} for i in indices.tolist()]`

---

## Summary

| Severity | Count | Description |
|----------|-------|-------------|
| **Critical** | 5 | Parameters not wired through (BUG-1, BUG-2, BUG-3), dead code that appears functional (BUG-4, BUG-5) |
| **Moderate** | 4 | VAPO NLL scaling (BUG-7), stats mismatch after filtering (BUG-8), GDPO grouping fragility (BUG-10), distillation never called (BUG-11, BUG-12) |
| **Low** | 6 | token_returns not implemented (BUG-13), dead config fields (BUG-14, BUG-18), no validation (BUG-15), performance (BUG-17), latent mutation bugs (BUG-19, BUG-20) |

### Priority Fix Order

1. **BUG-1** + **BUG-2**: Wire all new parameters through `create_training_engines` and `create_rollout_engines`. Without this, SAPO temperatures, M2PO threshold, VAPO, advantage mode, and KL mode are all silently ignored.
2. **BUG-3** + **BUG-4**: Wire `kl_mode` and integrate `AdaptiveKLController` into the training loop.
3. **BUG-5**: Implement constant denominator mode for Dr.GRPO.
4. **BUG-11** + **BUG-12**: Wire distillation parameters and integrate distillation loss into `train_step`.
5. **BUG-7**: Fix VAPO NLL loss scaling.
6. **BUG-13**: Implement token-level discounted returns for REINFORCE++.
7. **BUG-15**: Add config validation for all new fields.
