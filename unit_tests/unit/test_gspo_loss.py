import torch
import pytest
import numpy as np
from types import SimpleNamespace
from algs.GSPO.gspo import GSPO


def test_gspo_sequence_mean_ratio():
    """Verify the geometric mean ratio is computed correctly."""
    gspo_logic = GSPO
    dummy_self = SimpleNamespace(
        clip_low=0.2,
        clip_high=0.2,
        ent_coeff=0.0,
        kl_coeff=0.0,
        alg_name="Mock",
    )

    # logprobs - old_logprobs = [0.1, 0.2, 0.3], mean = 0.2
    logprobs = torch.tensor([[-0.4, -0.3, -0.2]])
    old_logprobs = torch.tensor([[-0.5, -0.5, -0.5]])
    advantages = torch.tensor([[1.0, 1.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])

    loss, denom, metrics = gspo_logic.compute_policy_loss(
        dummy_self, logprobs, old_logprobs, advantages, mask, None, None
    )

    # The combined_logratio uses mean_logratio = mean([0.1, 0.2, 0.3]) = 0.2
    # combined_ratio = exp(0.2) ≈ 1.2214 for all tokens
    # But clip_high=0.2 means max ratio is 1.2, so clipping applies:
    # clipped_ratio = min(exp(0.2), 1.2) = 1.2
    # Since unclipped*adv > clipped*adv when adv>0 and ratio>1+clip_high,
    # the PPO min takes the clipped version: pi_loss = -(1.2 * 1.0 * 3) / 3 = -1.2
    assert np.isclose(metrics['pi_loss'], -1.2, atol=1e-4)


def test_gspo_loss_identical_policy():
    """When logprobs == old_logprobs, ratio should be 1 everywhere."""
    gspo_logic = GSPO
    dummy_self = SimpleNamespace(
        clip_low=0.2,
        clip_high=0.2,
        ent_coeff=0.0,
        kl_coeff=0.0,
        alg_name="Mock",
    )

    logprobs = torch.tensor([[-0.5, -0.3, -0.7]])
    old_logprobs = torch.tensor([[-0.5, -0.3, -0.7]])
    advantages = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])

    loss, denom, metrics = gspo_logic.compute_policy_loss(
        dummy_self, logprobs, old_logprobs, advantages, mask, None, None
    )

    # ratio = 1.0 everywhere, so pi_loss = -(1*1 + 1*2 + 1*3)/3 = -2.0
    assert np.isclose(metrics['pi_loss'], -2.0, atol=1e-5)
    assert np.isclose(metrics['approx_kl'], 0.0, atol=1e-7)
    assert np.isclose(metrics['clipfrac'], 0.0, atol=1e-7)


def test_gspo_stop_gradient():
    """Verify gradients flow through per-token logprobs but not sequence ratio."""
    gspo_logic = GSPO
    dummy_self = SimpleNamespace(
        clip_low=0.2,
        clip_high=0.2,
        ent_coeff=0.0,
        kl_coeff=0.0,
        alg_name="Mock",
    )

    logprobs = torch.tensor([[-0.3, -0.5]], requires_grad=True)
    old_logprobs = torch.tensor([[-0.4, -0.6]])
    advantages = torch.tensor([[1.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0]])

    loss, denom, metrics = gspo_logic.compute_policy_loss(
        dummy_self, logprobs, old_logprobs, advantages, mask, None, None
    )
    loss.backward()

    # Gradients should exist and be non-zero (flow through per-token logprobs)
    assert logprobs.grad is not None
    assert not torch.isnan(logprobs.grad).any()
    assert logprobs.grad.abs().sum().item() > 0.0

    # The key property of GSPO: combined_logratio = logprobs - logprobs.detach() + mean_logratio.detach()
    # So the gradient of ratio w.r.t. logprobs is through the first term only,
    # meaning each token gets gradient proportional to exp(mean_logratio) * advantage,
    # NOT through the mean_logratio computation.
    # We verify this by checking the gradient is uniform across tokens with same advantage:
    # Both tokens have advantage=1.0, so gradients should be equal.
    assert torch.allclose(logprobs.grad[0, 0], logprobs.grad[0, 1], atol=1e-5)


def test_gspo_mask_respected():
    """Padded positions don't affect the sequence-level ratio."""
    gspo_logic = GSPO
    dummy_self = SimpleNamespace(
        clip_low=0.2,
        clip_high=0.2,
        ent_coeff=0.0,
        kl_coeff=0.0,
        alg_name="Mock",
    )

    # Sequence where last 2 tokens are padded
    logprobs = torch.tensor([[-0.3, -0.5, -999.0, -999.0]])
    old_logprobs = torch.tensor([[-0.3, -0.5, -0.1, -0.1]])
    advantages = torch.tensor([[1.0, 2.0, 100.0, 100.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    loss, denom, metrics = gspo_logic.compute_policy_loss(
        dummy_self, logprobs, old_logprobs, advantages, mask, None, None
    )

    # Only first 2 tokens contribute: logratio = [0.0, 0.0], mean = 0.0, ratio = 1.0
    # pi_loss = -(1*1 + 1*2)/2 = -1.5
    assert np.isclose(metrics['pi_loss'], -1.5, atol=1e-5)
    # The extreme logprobs at masked positions should not cause NaN or Inf
    assert np.isfinite(loss.item())
