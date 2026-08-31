from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_learner_tunnel_is_tls_with_http2() -> None:
    source = (ROOT / "src" / "armnet_rlt" / "modal_app.py").read_text()
    assert "modal.forward(LEARNER_PORT, h2_enabled=True)" in source
    assert "modal.forward(LEARNER_PORT, unencrypted=True)" not in source


def test_modal_learner_image_does_not_install_openpi_or_jax() -> None:
    source = (ROOT / "src" / "armnet_rlt" / "modal_app.py").read_text()
    image_recipe = source.split("image = (", 1)[1].split("app = modal.App", 1)[0]
    assert "github.com/pravsels/openpi" not in image_recipe
    assert "'jax[" not in image_recipe
    assert "torch==2.8.0" in image_recipe
