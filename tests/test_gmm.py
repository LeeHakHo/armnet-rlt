import math

import torch
import torch.nn as nn

from armnet_rlt.gmm import GMMHead, chunk_log_prob, chunk_weights, pick_chunk


def _mixture(batch=2, steps=4, modes=3, dim=2):
    torch.manual_seed(0)
    means = torch.randn(batch, steps, modes, dim)
    log_std = torch.full((batch, steps, modes, dim), math.log(0.1))
    logits = torch.zeros(batch, steps, modes)
    return means, log_std, logits


def test_head_unpacks_means_log_stds_and_logits() -> None:
    head = GMMHead(8, action_dim=2, num_modes=3, min_log_std=-2.0, max_log_std=1.0)
    raw = head(torch.randn(5, 4, 8))
    assert raw.shape == (5, 4, head.output_dim)
    means, log_std, logits = head.unpack(raw)
    assert means.shape == (5, 4, 3, 2) and log_std.shape == (5, 4, 3, 2) and logits.shape == (5, 4, 3)
    assert float(log_std.min()) >= -2.0 and float(log_std.max()) <= 1.0


def test_chunk_likelihood_favours_the_mode_that_matches_the_actions() -> None:
    means, log_std, logits = _mixture()
    actions = means[:, :, 1, :].clone()
    mixture_ll, mode_ll = chunk_log_prob(means, log_std, logits, actions)
    assert mode_ll.argmax(dim=-1).tolist() == [1, 1]
    # The mixture is at most the best mode (weight <= 1) and no worse than it
    # by more than log K.
    assert torch.all(mixture_ll <= mode_ll.max(dim=-1).values + 1e-6)
    assert torch.all(mixture_ll >= mode_ll.max(dim=-1).values - math.log(3) - 1e-6)
    # Padded steps do not count: garbage there leaves the likelihood unchanged.
    valid = torch.ones(2, 4, dtype=torch.bool)
    valid[:, 2:] = False
    clean, _ = chunk_log_prob(means, log_std, logits, actions, valid)
    garbage = actions.clone()
    garbage[:, 2:] = 100.0
    dirty, _ = chunk_log_prob(means, log_std, logits, garbage, valid)
    assert torch.allclose(clean, dirty)
    full, _ = chunk_log_prob(means, log_std, logits, garbage)
    assert torch.all(full < clean)


def test_pick_chunk_executes_the_likeliest_mode_not_the_mean() -> None:
    means, log_std, logits = _mixture()
    logits[:, :, 2] = 5.0
    chunk, chosen = pick_chunk(means, log_std, logits)
    assert chosen.tolist() == [2, 2]
    assert torch.equal(chunk, means[:, :, 2, :])
    assert not torch.allclose(chunk, means.mean(dim=2))
    weights = chunk_weights(logits)
    assert weights.argmax(dim=-1).tolist() == [2, 2] and torch.allclose(weights.sum(-1), torch.ones(2))
    torch.manual_seed(1)
    noisy, _ = pick_chunk(means, log_std, logits, sample_noise=True)
    assert not torch.equal(noisy, chunk) and torch.allclose(noisy, chunk, atol=0.6)


def test_modes_start_at_the_linear_head_and_separate() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(8, 2)
    head = GMMHead(8, action_dim=2, num_modes=3)
    head.init_from_linear(linear, init_std=0.2)
    features = torch.randn(4, 8)
    means, log_std, logits = head.unpack(head(features.unsqueeze(1)))
    reference = linear(features).unsqueeze(1)
    for mode in range(3):
        assert torch.allclose(means[:, :, mode, :], reference, atol=0.2)
    assert not torch.allclose(means[:, :, 0, :], means[:, :, 1, :])
    assert torch.allclose(log_std.exp(), torch.full_like(log_std, 0.2), atol=1e-5)
    assert torch.allclose(logits, torch.zeros_like(logits))
