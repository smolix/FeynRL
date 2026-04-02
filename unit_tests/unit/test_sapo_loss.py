import torch
import pytest
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock
from algs.SAPO.sapo import SAPO


def test_sapo_gate_function():
    """Verify gate(r=1.0, tau) = sigmoid(0) * (4/tau) = 0.5 * (4/tau) = 2/tau."""
    sapo_logic = SAPO
    dummy_self = SimpleNamespace(
        clip_low=0.2,
        clip_high=0.2,
        ent_coeff=0.0,
        kl_coeff=0.0,
        tau_pos=2.0,
        tau_neg=2.0,
        alg_name="Mock",
    )

    # ratio = exp(0) = 1.0, so gate = sigmoid(tau*(1-1)) * 4/tau = 0.5 * 4/tau = 2/tau
    logprobs = torch.tensor([[-0.5, -0.3]])
    old_logprobs = torch.tensor([[-0.5, -0.3]])
    advantages = torch.tensor([[1.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0]])

    loss, denom, metrics = sapo_logic.compute_policy_loss(
        dummy_self, logprobs, old_logprobs, advantages, mask, None, None
    )

    # gate = 2/tau = 2/2 = 1.0
    # pi_loss = -(1.0 * 1.0 + 1.0 * 1.0) / 2 = -1.0
    assert np.isclose(metrics['pi_loss'], -1.0, atol=1e-5)


def test_sapo_loss_zero_advantage():
    """When advantages are all zero, the pi_loss should be zero."""
    sapo_logic = SAPO
    dummy_self = SimpleNamespace(
        clip_low=0.2,
        clip_high=0.2,
        ent_coeff=0.0,
        kl_coeff=0.0,
        tau_pos=1.0,
        tau_neg=1.05,
        alg_name="Mock",
    )

    logprobs = torch.tensor([[-0.3, -0.5, -0.2]])
    old_logprobs = torch.tensor([[-0.4, -0.5, -0.1]])
    advantages = torch.tensor([[0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])

    loss, denom, metrics = sapo_logic.compute_policy_loss(
        dummy_self, logprobs, old_logprobs, advantages, mask, None, None
    )

    assert np.isclose(metrics['pi_loss'], 0.0, atol=1e-7)
    assert np.isclose(loss.item(), 0.0, atol=1e-7)


def test_sapo_loss_positive_advantage():
    """With positive advantages, loss should be negative (encouraging higher logprob)."""
    sapo_logic = SAPO
    dummy_self = SimpleNamespace(
        clip_low=0.2,
        clip_high=0.2,
        ent_coeff=0.0,
        kl_coeff=0.0,
        tau_pos=1.0,
        tau_neg=1.05,
        alg_name="Mock",
    )

    logprobs = torch.tensor([[-0.3, -0.5]])
    old_logprobs = torch.tensor([[-0.3, -0.5]])
    advantages = torch.tensor([[2.0, 3.0]])
    mask = torch.tensor([[1.0, 1.0]])

    loss, denom, metrics = sapo_logic.compute_policy_loss(
        dummy_self, logprobs, old_logprobs, advantages, mask, None, None
    )

    # With ratio=1.0 and positive advantages, pi_loss should be negative
    assert metrics['pi_loss'] < 0.0


def test_sapo_different_temperatures():
    """Verify tau_pos and tau_neg produce different gates for the same ratio."""
    sapo_logic = SAPO
    # Use very different tau values
    dummy_self_sym = SimpleNamespace(
        clip_low=0.2,
        clip_high=0.2,
        ent_coeff=0.0,
        kl_coeff=0.0,
        tau_pos=1.0,
        tau_neg=1.0,
        alg_name="Mock",
    )
    dummy_self_asym = SimpleNamespace(
        clip_low=0.2,
        clip_high=0.2,
        ent_coeff=0.0,
        kl_coeff=0.0,
        tau_pos=1.0,
        tau_neg=3.0,
        alg_name="Mock",
    )

    # ratio = exp(0.5) > 1, so gate uses tau_pos for positive adv, tau_neg for negative adv
    logprobs = torch.tensor([[-0.1, -0.1]])
    old_logprobs = torch.tensor([[-0.6, -0.6]])
    # First token: positive advantage (uses tau_pos), second: negative (uses tau_neg)
    advantages = torch.tensor([[1.0, -1.0]])
    mask = torch.tensor([[1.0, 1.0]])

    _, _, metrics_sym = sapo_logic.compute_policy_loss(
        dummy_self_sym, logprobs, old_logprobs, advantages, mask, None, None
    )
    _, _, metrics_asym = sapo_logic.compute_policy_loss(
        dummy_self_asym, logprobs, old_logprobs, advantages, mask, None, None
    )

    # Different tau_neg should give different pi_loss
    assert not np.isclose(metrics_sym['pi_loss'], metrics_asym['pi_loss'], atol=1e-4)


def test_sapo_mask_respected():
    """Padded positions should not contribute to loss."""
    sapo_logic = SAPO
    dummy_self = SimpleNamespace(
        clip_low=0.2,
        clip_high=0.2,
        ent_coeff=0.0,
        kl_coeff=0.0,
        tau_pos=1.0,
        tau_neg=1.05,
        alg_name="Mock",
    )

    logprobs = torch.tensor([[-0.3, -0.5, -0.2, -0.1]])
    old_logprobs = torch.tensor([[-0.3, -0.5, -0.2, -0.1]])
    advantages = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask_full = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    mask_partial = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    _, _, metrics_full = sapo_logic.compute_policy_loss(
        dummy_self, logprobs, old_logprobs, advantages, mask_full, None, None
    )
    _, _, metrics_partial = sapo_logic.compute_policy_loss(
        dummy_self, logprobs, old_logprobs, advantages, mask_partial, None, None
    )

    # With mask_partial, only first 2 tokens contribute
    # The pi_loss per token should be different because different advantages are included
    # Partial should only see advantages [1.0, 2.0], full sees [1.0, 2.0, 3.0, 4.0]
    assert not np.isclose(metrics_full['pi_loss'], metrics_partial['pi_loss'], atol=1e-5)

    # Verify: with ratio=1 and tau_pos=1.0, gate = 2/1 = 2.0
    # partial: -(2.0*1.0 + 2.0*2.0)/2 = -(2+4)/2 = -3.0
    # full:    -(2.0*1.0 + 2.0*2.0 + 2.0*3.0 + 2.0*4.0)/4 = -(2+4+6+8)/4 = -5.0
    assert np.isclose(metrics_partial['pi_loss'], -3.0, atol=1e-4)
    assert np.isclose(metrics_full['pi_loss'], -5.0, atol=1e-4)
