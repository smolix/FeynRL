import torch
import pytest
import numpy as np
from rollouts.replay_buffer import ReplayBuffer


def _make_buffer_with_items(rewards_and_masks):
    """
    Create a ReplayBuffer with items that have 'rewards' and 'masks' tensors.
    rewards_and_masks: list of (reward_sum, mask_count) tuples.
    """
    rb = ReplayBuffer(pad_token_id=0, max_seq_len=10)
    for reward_sum, mask_count in rewards_and_masks:
        # Create a reward tensor where the sum equals reward_sum
        rewards = torch.zeros(10, dtype=torch.float32)
        rewards[-1] = reward_sum
        # Create a mask tensor with mask_count valid tokens
        masks = torch.zeros(10, dtype=torch.float32)
        masks[:mask_count] = 1.0
        rb.items.append({'rewards': rewards, 'masks': masks})
    rb.total_action_tokens = sum(mc for _, mc in rewards_and_masks)
    return rb


def test_resample_preserves_count():
    """Resampling should keep the same number of items."""
    rb = _make_buffer_with_items([
        (1.0, 3), (2.0, 4), (3.0, 5), (0.5, 2), (4.0, 6)
    ])
    original_count = len(rb.items)

    result = rb.resample_by_reward()

    assert result == original_count
    assert len(rb.items) == original_count


def test_resample_favors_high_reward():
    """Items with higher |reward| should appear more frequently on average."""
    torch.manual_seed(42)

    rb = _make_buffer_with_items([
        (0.01, 3),  # Very low reward
        (0.01, 3),  # Very low reward
        (0.01, 3),  # Very low reward
        (10.0, 5),  # Very high reward
    ])

    # Resample many times and count how often the high-reward item appears
    high_reward_count = 0
    n_trials = 200
    for trial in range(n_trials):
        torch.manual_seed(trial)
        rb_copy = _make_buffer_with_items([
            (0.01, 3), (0.01, 3), (0.01, 3), (10.0, 5)
        ])
        rb_copy.resample_by_reward()
        # Count items with high reward sum
        for item in rb_copy.items:
            if item['rewards'].sum().item() > 5.0:
                high_reward_count += 1

    # The high-reward item (index 3) has |10|^2 = 100 weight vs |0.01|^2 = 0.0001 each
    # So it should dominate the resampling
    avg_high = high_reward_count / n_trials
    assert avg_high > 3.0, f"High-reward item appeared {avg_high} times on average (expected ~4)"


def test_resample_empty_buffer():
    """Empty buffer should return 0."""
    rb = ReplayBuffer(pad_token_id=0, max_seq_len=10)

    result = rb.resample_by_reward()

    assert result == 0
    assert len(rb.items) == 0


def test_resample_recounts_tokens():
    """total_action_tokens should be recalculated after resampling."""
    rb = _make_buffer_with_items([
        (1.0, 2),   # 2 valid tokens
        (5.0, 8),   # 8 valid tokens
    ])
    original_tokens = rb.total_action_tokens
    assert original_tokens == 10  # 2 + 8

    torch.manual_seed(0)
    rb.resample_by_reward()

    # After resampling, total_action_tokens should be recalculated based on
    # the actual masks of the resampled items
    expected_tokens = sum(int((item['masks'] > 0.5).sum().item()) for item in rb.items)
    assert rb.total_action_tokens == expected_tokens
    # It should still be a valid positive number
    assert rb.total_action_tokens > 0
