import torch
import pytest
import numpy as np
from types import SimpleNamespace
from algs.M2PO.m2po import M2PO


def test_m2po_no_masking_small_divergence():
    """When all log-ratios are small (< threshold), no tokens should be masked."""
    m2po_logic = M2PO
    dummy_self = SimpleNamespace(
        clip_low=0.2,
        clip_high=0.2,
        ent_coeff=0.0,
        kl_coeff=0.0,
        m2_threshold=0.2,
        alg_name="Mock",
    )

    # Very small difference: delta^2 will be tiny (all << 0.2)
    logprobs = torch.tensor([[-0.50, -0.51, -0.49, -0.50]])
    old_logprobs = torch.tensor([[-0.50, -0.50, -0.50, -0.50]])
    advantages = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

    loss, denom, metrics = m2po_logic.compute_policy_loss(
        dummy_self, logprobs, old_logprobs, advantages, mask, None, None
    )

    # No tokens should be masked since delta^2 values are all tiny
    assert np.isclose(metrics['m2_mask_ratio'], 0.0, atol=1e-5)
    # denom should be 4 (all tokens valid)
    assert np.isclose(denom.item(), 4.0)


def test_m2po_masks_high_divergence():
    """When some tokens have very high log-ratio divergence, they should be masked."""
    m2po_logic = M2PO
    dummy_self = SimpleNamespace(
        clip_low=0.2,
        clip_high=0.2,
        ent_coeff=0.0,
        kl_coeff=0.0,
        m2_threshold=0.01,  # Very low threshold to trigger masking
        alg_name="Mock",
    )

    # Token 0: delta=5.0, delta^2=25.0 (very high, should be masked)
    # Tokens 1-3: delta~0, delta^2~0 (should pass)
    logprobs = torch.tensor([[-0.5, -0.5, -0.5, -0.5]])
    old_logprobs = torch.tensor([[4.5, -0.5, -0.5, -0.5]])
    advantages = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

    loss, denom, metrics = m2po_logic.compute_policy_loss(
        dummy_self, logprobs, old_logprobs, advantages, mask, None, None
    )

    # At least one token should be masked (the one with delta=5.0)
    assert metrics['m2_mask_ratio'] > 0.0
    # denom should be less than 4 (some tokens masked)
    assert denom.item() < 4.0


def test_m2po_mask_ratio_metric():
    """The m2_mask_ratio metric should reflect the fraction of tokens masked."""
    m2po_logic = M2PO
    dummy_self = SimpleNamespace(
        clip_low=0.2,
        clip_high=0.2,
        ent_coeff=0.0,
        kl_coeff=0.0,
        m2_threshold=0.01,  # Very low threshold
        alg_name="Mock",
    )

    # 2 tokens with huge divergence, 2 with zero divergence
    logprobs = torch.tensor([[-0.5, -0.5, -0.5, -0.5]])
    old_logprobs = torch.tensor([[2.5, 2.5, -0.5, -0.5]])
    advantages = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

    loss, denom, metrics = m2po_logic.compute_policy_loss(
        dummy_self, logprobs, old_logprobs, advantages, mask, None, None
    )

    # m2_mask_ratio should be between 0 and 1
    assert 0.0 <= metrics['m2_mask_ratio'] <= 1.0
    # With 2 high-divergence tokens out of 4, expect some masking
    assert metrics['m2_mask_ratio'] > 0.0


def test_m2po_all_identical():
    """When logprobs == old_logprobs, no tokens should be masked."""
    m2po_logic = M2PO
    dummy_self = SimpleNamespace(
        clip_low=0.2,
        clip_high=0.2,
        ent_coeff=0.0,
        kl_coeff=0.0,
        m2_threshold=0.2,
        alg_name="Mock",
    )

    logprobs = torch.tensor([[-0.3, -0.5, -0.7, -0.2]])
    old_logprobs = torch.tensor([[-0.3, -0.5, -0.7, -0.2]])
    advantages = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

    loss, denom, metrics = m2po_logic.compute_policy_loss(
        dummy_self, logprobs, old_logprobs, advantages, mask, None, None
    )

    # delta = 0 for all tokens, so m2 = 0, nothing should be masked
    assert np.isclose(metrics['m2_mask_ratio'], 0.0, atol=1e-7)
    # denom should equal total valid tokens
    assert np.isclose(denom.item(), 4.0)
    # Loss should be standard PPO with ratio=1: -(1+2+3+4)/4 = -2.5
    assert np.isclose(metrics['pi_loss'], -2.5, atol=1e-5)
