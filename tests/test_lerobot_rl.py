import numpy as np
import torch

from armnet_rlt.lerobot_rl import CHUNK_LEN, RL_TOKEN_DIM, RLTokenModel, _full_noise


def test_rl_token_has_the_actor_width_and_respects_the_mask() -> None:
    torch.manual_seed(0)
    model = RLTokenModel(in_dim=16, width=32, heads=4, mlp_dim=64)
    emb = torch.randn(3, 5, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]], dtype=torch.bool)
    token = model.encode(emb, mask)
    assert token.shape == (3, 32)
    # Padded positions do not change the token.
    changed = emb.clone()
    changed[0, 3:] = 100.0
    torch.testing.assert_close(model.encode(changed, mask)[0], token[0])
    assert RLTokenModel(in_dim=8).config["width"] == RL_TOKEN_DIM


def test_rl_token_learns_to_reconstruct_and_round_trips(tmp_path) -> None:
    torch.manual_seed(0)
    model = RLTokenModel(in_dim=8, width=32, heads=4, mlp_dim=64)
    data = torch.randn(64, 4, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = float(model.reconstruction_loss(data))
    for _ in range(200):
        loss = model.reconstruction_loss(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(loss) < 0.5 * first
    model.save(tmp_path)
    loaded = RLTokenModel.load(tmp_path)
    torch.testing.assert_close(loaded.encode(data[:2]), model.eval().encode(data[:2]))


def test_full_noise_repeats_the_last_row_and_zero_pads() -> None:
    noise = torch.arange(CHUNK_LEN * 6, dtype=torch.float32).reshape(1, CHUNK_LEN, 6)
    full = _full_noise(noise, 50, 32)
    assert full.shape == (1, 50, 32)
    torch.testing.assert_close(full[0, :CHUNK_LEN, :6], noise[0])
    torch.testing.assert_close(full[0, CHUNK_LEN:, :6], noise[0, -1].expand(50 - CHUNK_LEN, 6))
    assert float(full[..., 6:].abs().sum()) == 0.0
    np.testing.assert_array_equal(_full_noise(noise, 20, 6).numpy(), noise[:, :20].numpy())
