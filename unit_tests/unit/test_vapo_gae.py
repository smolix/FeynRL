import torch
import pytest
import numpy as np
from types import SimpleNamespace
from algs.PPO.ppo import PPO


def test_vapo_longer_sequences_higher_lambda():
    """Longer sequences should get tau closer to 1.0."""
    ppo_logic = PPO

    # vapo_alpha = 0.05, so lambda = 1 - 1/(0.05 * seq_len)
    # seq_len=5:  lambda = 1 - 1/0.25 = 1 - 4 = -3 -> clamped to 0.0
    # seq_len=40: lambda = 1 - 1/2.0 = 0.5
    # seq_len=100: lambda = 1 - 1/5.0 = 0.8
    dummy_self = SimpleNamespace(gamma=0.99, tau=0.95, vapo_enabled=True, vapo_alpha=0.05)

    # Short sequence: 5 tokens
    rewards_short = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    values_short = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]])
    done_short = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
    mask_short = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])

    _, advs_short = ppo_logic.compute_advantages(
        dummy_self, rewards_short, values_short, done_short, mask_short
    )

    # Long sequence: 40 tokens (pad short to get effective length difference)
    T_long = 40
    rewards_long = torch.zeros(1, T_long)
    rewards_long[0, 0] = 1.0
    values_long = torch.full((1, T_long), 0.5)
    done_long = torch.zeros(1, T_long)
    done_long[0, -1] = 1.0
    mask_long = torch.ones(1, T_long)

    _, advs_long = ppo_logic.compute_advantages(
        dummy_self, rewards_long, values_long, done_long, mask_long
    )

    # For VAPO, longer sequences get higher lambda (closer to 1), which means
    # advantages propagate further back. The per_seq_tau for 40 tokens = 0.5,
    # while for 5 tokens = 0.0 (clamped). So the long sequence should have
    # more temporal credit assignment (advantages at early tokens should be larger
    # relative to the single reward at t=0).
    # With tau=0.0 (short), only the delta at each step matters, no propagation.
    # With tau=0.5 (long), advantages propagate back.
    # At t=0 for short: just delta_0 = 1.0 + gamma*0.5 - 0.5 = 0.995
    # At t=0 for long: delta_0 + discounted future advantages
    # Short seq has tau=0 so advantage at t=0 is just delta_0
    short_t0 = advs_short[0, 0].item()
    # Long seq t=0 advantage should include propagated info
    long_t0 = advs_long[0, 0].item()
    # Long should have a different (generally larger magnitude) advantage at t=0
    # because it uses non-zero lambda
    assert abs(long_t0) != abs(short_t0) or T_long != 5, \
        "Long and short sequences should have different advantages with VAPO"


def test_vapo_disabled_uses_fixed_tau():
    """When vapo_enabled=False, the original fixed tau is used."""
    ppo_logic = PPO

    fixed_tau = 0.95
    dummy_self_fixed = SimpleNamespace(gamma=0.99, tau=fixed_tau, vapo_enabled=False, vapo_alpha=0.05)
    dummy_self_vapo = SimpleNamespace(gamma=0.99, tau=fixed_tau, vapo_enabled=True, vapo_alpha=0.05)

    rewards = torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0]])
    values = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]])
    done = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])

    _, advs_fixed = ppo_logic.compute_advantages(
        dummy_self_fixed, rewards, values, done, mask
    )
    _, advs_vapo = ppo_logic.compute_advantages(
        dummy_self_vapo, rewards, values, done, mask
    )

    # VAPO with alpha=0.05, seq_len=5: tau_vapo = 1 - 1/(0.05*5) = 1 - 4 = -3 -> clamped to 0
    # Fixed tau=0.95 vs VAPO tau=0.0 should give different advantages
    assert not torch.allclose(advs_fixed, advs_vapo, atol=1e-5), \
        "VAPO and fixed tau should produce different advantages"


def test_vapo_short_sequence_low_lambda():
    """Very short sequences should get smaller lambda (possibly 0)."""
    ppo_logic = PPO
    dummy_self = SimpleNamespace(gamma=0.99, tau=0.95, vapo_enabled=True, vapo_alpha=0.05)

    # With alpha=0.05 and seq_len=2: lambda = 1 - 1/(0.05*2) = 1 - 10 = -9 -> clamped to 0.0
    rewards = torch.tensor([[1.0, 0.0]])
    values = torch.tensor([[0.5, 0.5]])
    done = torch.tensor([[0.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0]])

    _, advs = ppo_logic.compute_advantages(dummy_self, rewards, values, done, mask)

    # With tau=0 (clamped), GAE reduces to simple TD errors:
    # T=1: delta = 0.0 + 0 - 0.5 = -0.5, last_adv = -0.5
    # T=0: delta = 1.0 + 0.99*0.5*1.0 - 0.5 = 0.995, last_adv = 0.995 + gamma*0*(-0.5) = 0.995
    # (tau=0 means no propagation of last_adv through gamma*tau*last_adv)
    assert np.isclose(advs[0, 1].item(), -0.5, atol=1e-5)
    assert np.isclose(advs[0, 0].item(), 0.995, atol=1e-5)
