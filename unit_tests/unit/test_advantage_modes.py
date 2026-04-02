import sys
from unittest.mock import MagicMock

# Mock vllm before importing rollouts.vllm_engine
if "vllm" not in sys.modules:
    sys.modules["vllm"] = MagicMock()

import torch
import pytest
import numpy as np
from types import SimpleNamespace
from rollouts.vllm_engine import VLLMRolloutEngine


def _make_engine(advantage_mode, reward_broadcast=False, eps_reward_norm=1e-8):
    """Create a minimal SimpleNamespace that mimics VLLMRolloutEngine for normalize_rewards."""
    dummy = SimpleNamespace(
        advantage_mode=advantage_mode,
        reward_broadcast=reward_broadcast,
        eps_reward_norm=eps_reward_norm,
    )
    # Bind the unbound method so we can call it on our dummy
    dummy.normalize_rewards = lambda samples, stats, prompt_len, is_per_token: \
        VLLMRolloutEngine.normalize_rewards(dummy, samples, stats, prompt_len, is_per_token)
    return dummy


def _make_samples(rewards_list, seq_len=10, prompt_len=3):
    """Create sample dicts with token_rewards tensors matching the rollout engine format."""
    samples = []
    for r in rewards_list:
        token_rewards = torch.zeros(seq_len, dtype=torch.float32)
        token_rewards[-1] = r
        samples.append({'token_rewards': token_rewards})
    return samples


def test_zscore_normalization():
    """Default GRPO mode produces (r - mean) / std."""
    engine = _make_engine("zscore")
    rewards = [1.0, 3.0, 5.0, 7.0]
    samples = _make_samples(rewards)
    stats = {'rewards': rewards}

    engine.normalize_rewards(samples, stats, prompt_len=3, is_per_token=False)

    mean_r = np.mean(rewards)
    # Bessel's correction: std = sqrt(sum((r-mean)^2) / (n-1))
    std_r = np.sqrt(np.sum((np.array(rewards) - mean_r)**2) / (len(rewards) - 1))

    for i, s in enumerate(samples):
        expected = (rewards[i] - mean_r) / (std_r + 1e-8)
        actual = s['token_zscores'][-1].item()
        assert np.isclose(actual, expected, atol=1e-5), \
            f"Sample {i}: expected {expected}, got {actual}"


def test_mean_only_normalization():
    """Dr.GRPO mode produces (r - mean) without std division."""
    engine = _make_engine("mean_only")
    rewards = [1.0, 3.0, 5.0, 7.0]
    samples = _make_samples(rewards)
    stats = {'rewards': rewards}

    engine.normalize_rewards(samples, stats, prompt_len=3, is_per_token=False)

    mean_r = np.mean(rewards)

    for i, s in enumerate(samples):
        expected = rewards[i] - mean_r
        actual = s['token_zscores'][-1].item()
        assert np.isclose(actual, expected, atol=1e-5), \
            f"Sample {i}: expected {expected}, got {actual}"


def test_rloo_normalization():
    """RLOO mode produces G/(G-1) * (r - mean)."""
    engine = _make_engine("rloo")
    rewards = [1.0, 3.0, 5.0, 7.0]
    G = len(rewards)
    samples = _make_samples(rewards)
    stats = {'rewards': rewards}

    engine.normalize_rewards(samples, stats, prompt_len=3, is_per_token=False)

    mean_r = np.mean(rewards)

    for i, s in enumerate(samples):
        # A_i = G/(G-1) * (r_i - mean_all)
        expected = G / (G - 1) * (rewards[i] - mean_r)
        actual = s['token_zscores'][-1].item()
        assert np.isclose(actual, expected, atol=1e-5), \
            f"Sample {i}: expected {expected}, got {actual}"


def test_single_sample_modes():
    """With n_samples=1, all modes should handle gracefully without division by zero."""
    for mode in ["zscore", "mean_only", "rloo"]:
        engine = _make_engine(mode)
        rewards = [5.0]
        samples = _make_samples(rewards)
        stats = {'rewards': rewards}

        engine.normalize_rewards(samples, stats, prompt_len=3, is_per_token=False)

        val = samples[0]['token_zscores'][-1].item()
        assert np.isfinite(val), f"Mode {mode} produced non-finite value with single sample"

        if mode == "zscore":
            # With n=1, mean=0, std=1 (fallback), so zscore = (5-0)/(1+eps) = 5.0
            assert np.isclose(val, 5.0, atol=1e-5)
        elif mode == "mean_only":
            # mean=0 fallback, so result = 5 - 0 = 5
            assert np.isclose(val, 5.0, atol=1e-5)
        elif mode == "rloo":
            # With n=1, fallback returns raw reward
            assert np.isclose(val, 5.0, atol=1e-5)
