import torch
import pytest
import numpy as np
from types import SimpleNamespace
from algs.RL.common import COMMON

common_logic = COMMON


def test_kl_k1_simple():
    """k1 mode returns logprobs - ref_logprobs."""
    dummy_self = SimpleNamespace(_kl_mode='k1')
    logprobs = torch.tensor([[-0.1, -0.3, -0.5]])
    ref_logprobs = torch.tensor([[-0.2, -0.4, -0.3]])

    kl = common_logic.compute_kl_distance(dummy_self, logprobs, ref_logprobs, kl_mode='k1')

    expected = logprobs.float() - ref_logprobs.float()
    torch.testing.assert_close(kl, expected, atol=1e-6, rtol=1e-6)


def test_kl_k2_mse():
    """k2 mode returns 0.5 * (logprobs - ref_logprobs)^2."""
    dummy_self = SimpleNamespace(_kl_mode='k2')
    logprobs = torch.tensor([[-0.1, -0.3, -0.5]])
    ref_logprobs = torch.tensor([[-0.2, -0.4, -0.3]])

    kl = common_logic.compute_kl_distance(dummy_self, logprobs, ref_logprobs, kl_mode='k2')

    log_ratio = logprobs.float() - ref_logprobs.float()
    expected = 0.5 * log_ratio.pow(2)
    torch.testing.assert_close(kl, expected, atol=1e-6, rtol=1e-6)


def test_kl_k3_low_var():
    """k3 mode returns the variance-reduced form: exp(ref - pi) - (ref - pi) - 1 = exp(-lr) + lr - 1."""
    dummy_self = SimpleNamespace(_kl_mode='k3')
    logprobs = torch.tensor([[-0.1, -0.3]])
    ref_logprobs = torch.tensor([[-0.2, -0.5]])

    kl = common_logic.compute_kl_distance(dummy_self, logprobs, ref_logprobs, kl_mode='k3')

    log_ratio = logprobs.float() - ref_logprobs.float()
    expected = log_ratio + torch.exp(-log_ratio) - 1.0
    torch.testing.assert_close(kl, expected, atol=1e-6, rtol=1e-6)


def test_kl_abs():
    """abs mode returns |logprobs - ref_logprobs|."""
    dummy_self = SimpleNamespace()
    logprobs = torch.tensor([[-0.1, -0.5]])
    ref_logprobs = torch.tensor([[-0.3, -0.2]])

    kl = common_logic.compute_kl_distance(dummy_self, logprobs, ref_logprobs, kl_mode='abs')

    expected = (logprobs.float() - ref_logprobs.float()).abs()
    torch.testing.assert_close(kl, expected, atol=1e-6, rtol=1e-6)


def test_kl_identical_policies():
    """All modes should return ~0 when policies are identical."""
    dummy_self = SimpleNamespace()
    logprobs = torch.tensor([[-0.3, -0.5, -0.7]])
    ref_logprobs = logprobs.clone()

    for mode in ['k1', 'k2', 'k3', 'abs', 'k3_plus']:
        kl = common_logic.compute_kl_distance(dummy_self, logprobs, ref_logprobs, kl_mode=mode)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-7), \
            f"Mode {mode} did not return ~0 for identical policies: {kl}"


def test_kl_k3_plus_straight_through():
    """k3+ should have same forward value as k3 but different gradient (k2 gradient)."""
    dummy_self = SimpleNamespace()
    logprobs_k3 = torch.tensor([[-0.1, -0.5]], requires_grad=True)
    ref_logprobs = torch.tensor([[-0.3, -0.2]])

    logprobs_k3p = logprobs_k3.detach().clone().requires_grad_(True)

    # Forward values should match
    kl_k3 = common_logic.compute_kl_distance(dummy_self, logprobs_k3, ref_logprobs, kl_mode='k3')
    kl_k3p = common_logic.compute_kl_distance(dummy_self, logprobs_k3p, ref_logprobs, kl_mode='k3_plus')
    torch.testing.assert_close(kl_k3.detach(), kl_k3p.detach(), atol=1e-6, rtol=1e-6)

    # Backward: k3+ should use k2 gradient, which differs from k3 gradient
    kl_k3.sum().backward()
    kl_k3p.sum().backward()

    # k3 gradient: 1 - exp(ref - pi) = 1 - exp(-log_ratio)
    # k2 gradient: log_ratio = logprobs - ref_logprobs
    # These should NOT be equal (unless log_ratio happens to be a specific value)
    assert logprobs_k3.grad is not None
    assert logprobs_k3p.grad is not None
    # For logprobs_k3 = [-0.1, -0.5], ref = [-0.3, -0.2]:
    # log_ratio = [0.2, -0.3]
    # k3 grad: 1 - exp(-0.2) = 0.1813, 1 - exp(0.3) = -0.3499
    # k2 grad: 0.2, -0.3
    assert not torch.allclose(logprobs_k3.grad, logprobs_k3p.grad, atol=1e-3), \
        "k3 and k3+ should have different gradients"
