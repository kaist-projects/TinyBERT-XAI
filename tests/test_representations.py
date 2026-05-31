import torch

from tinybert_xai.analysis.representations import _linear_cka, _masked_row_kl


def test_linear_cka_is_one_for_scaled_copies():
    x = torch.randn(64, 8)
    y = 3.0 * x  # linear CKA is invariant to isotropic scaling
    assert _linear_cka(x, y) == torch.tensor(1.0).item() or abs(_linear_cka(x, y) - 1.0) < 1e-6


def test_linear_cka_is_near_zero_for_independent_features():
    torch.manual_seed(0)
    x = torch.randn(2048, 16)
    y = torch.randn(2048, 16)
    assert _linear_cka(x, y) < 0.05


def test_masked_row_kl_is_zero_for_identical_maps():
    attn = torch.softmax(torch.randn(2, 5, 5), dim=-1)
    mask = torch.ones(2, 5)
    total, count = _masked_row_kl(attn, attn.clone(), mask)
    assert count == 10
    assert abs(total) < 1e-5


def test_masked_row_kl_counts_only_valid_queries():
    attn = torch.softmax(torch.randn(1, 4, 4), dim=-1)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    _, count = _masked_row_kl(attn, attn.clone(), mask)
    assert count == 2
