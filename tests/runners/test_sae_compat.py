"""Tests for the SAE compatibility shim."""

from __future__ import annotations

from sae_lens import JumpReLUSAE, JumpReLUSAEConfig
from sae_lens.saes.sae import SAEMetadata

from saebench_audit.runners.sae_compat import patch_sae


def test_patch_sae_attaches_hook_name_and_layer() -> None:
    cfg = JumpReLUSAEConfig(
        d_in=4,
        d_sae=8,
        device="cpu",
        dtype="float32",
        metadata=SAEMetadata(hook_name="blocks.3.hook_resid_post", model_name="gpt2"),
    )
    sae = JumpReLUSAE(cfg)
    patched = patch_sae(sae)
    assert patched.cfg.hook_name == "blocks.3.hook_resid_post"
    assert patched.cfg.hook_layer == 3


def test_patch_sae_handles_missing_metadata_gracefully() -> None:
    cfg = JumpReLUSAEConfig(d_in=4, d_sae=8, device="cpu", dtype="float32")
    sae = JumpReLUSAE(cfg)
    # cfg.metadata exists by default but hook_name may be None / unset.
    patched = patch_sae(sae)
    assert patched is sae  # no-op return


def test_patch_sae_falls_back_to_layer_0_when_pattern_missing() -> None:
    cfg = JumpReLUSAEConfig(
        d_in=4,
        d_sae=8,
        device="cpu",
        dtype="float32",
        metadata=SAEMetadata(hook_name="model.layers.7", model_name="gpt2"),
    )
    sae = JumpReLUSAE(cfg)
    patched = patch_sae(sae)
    # The regex looks for "blocks.<n>." — this string doesn't match it.
    assert patched.cfg.hook_layer == 0
