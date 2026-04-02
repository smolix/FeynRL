import torch
import pytest
import numpy as np
from algs.RL.common import AdaptiveKLController


def test_adaptive_kl_increases_when_above_target():
    """When current_kl > target, value should increase."""
    controller = AdaptiveKLController(init_kl_coef=0.1, target_kl=0.01, horizon=10000)
    initial_value = controller.value

    controller.update(current_kl=0.05, n_steps=1)

    assert controller.value > initial_value


def test_adaptive_kl_decreases_when_below_target():
    """When current_kl < target, value should decrease."""
    controller = AdaptiveKLController(init_kl_coef=0.1, target_kl=0.1, horizon=10000)
    initial_value = controller.value

    controller.update(current_kl=0.01, n_steps=1)

    assert controller.value < initial_value


def test_adaptive_kl_stable_at_target():
    """When current_kl == target, value should stay roughly the same."""
    controller = AdaptiveKLController(init_kl_coef=0.1, target_kl=0.05, horizon=10000)
    initial_value = controller.value

    controller.update(current_kl=0.05, n_steps=1)

    # proportional_error = clip(0.05/0.05 - 1, -0.2, 0.2) = 0
    # mult = 1 + 0 * 1/10000 = 1.0
    assert np.isclose(controller.value, initial_value, atol=1e-10)


def test_adaptive_kl_minimum_bound():
    """Value should never go below 1e-8."""
    controller = AdaptiveKLController(init_kl_coef=1e-7, target_kl=1.0, horizon=100)

    # Repeatedly update with very low KL to push value down
    for _ in range(1000):
        controller.update(current_kl=0.0001, n_steps=10)

    assert controller.value >= 1e-8


def test_adaptive_kl_proportional_error_clipped():
    """The proportional error should be clipped to [-0.2, 0.2]."""
    # Even with a huge KL overshoot, the update should be bounded
    controller = AdaptiveKLController(init_kl_coef=0.1, target_kl=0.001, horizon=10000)
    initial_value = controller.value

    # current_kl/target = 100/0.001 = 100000 >> 1, but clipped to 0.2
    controller.update(current_kl=100.0, n_steps=1)

    # mult = 1 + 0.2 * 1/10000 = 1.00002
    expected = initial_value * (1.0 + 0.2 * 1 / 10000)
    assert np.isclose(controller.value, expected, atol=1e-10)


def test_adaptive_kl_n_steps_scaling():
    """Larger n_steps should produce a bigger adjustment."""
    ctrl_small = AdaptiveKLController(init_kl_coef=0.1, target_kl=0.01, horizon=10000)
    ctrl_large = AdaptiveKLController(init_kl_coef=0.1, target_kl=0.01, horizon=10000)

    ctrl_small.update(current_kl=0.05, n_steps=1)
    ctrl_large.update(current_kl=0.05, n_steps=100)

    # Both should increase, but large n_steps should increase more
    assert ctrl_large.value > ctrl_small.value
