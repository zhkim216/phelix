"""
Smoke test for the context / encoder edge-update refactor in atom_mpnn.py.

Covers:
  (1) ContextModule forward/backward shape + grad flow under both
      context_feature_edge_update = False / True.
  (2) EncLayer forward/backward under is_last_layer = False / True
      (exercises the encoder_edge_update flag path).

Run from the repo root:
    python scripts/tests/smoke_test_context_edge_update.py
"""
from __future__ import annotations

import torch

from allatom_design.model.seq_denoiser.denoisers.seq_design.atom_mpnn import (
    ContextModule,
    EncLayer,
)


def _check_grad_for_keywords(module: torch.nn.Module, required_keywords: list[str]) -> None:
    """Assert that every parameter whose name contains any required keyword received a non-zero grad."""
    for name, param in module.named_parameters():
        if any(kw in name for kw in required_keywords):
            assert param.grad is not None, f"no grad tensor on {name}"
            assert param.grad.abs().sum() > 0, f"zero grad on {name}"


def test_context_module(context_feature_edge_update: bool) -> None:
    torch.manual_seed(0)
    B, L, M, C = 2, 6, 4, 32  # batch, protein tokens, ligand atoms, hidden

    module = ContextModule(
        hidden_dim=C,
        dropout_p=0.1,
        num_processor_layers=2,
        num_aggregator_layers=2,
        context_feature_edge_update=context_feature_edge_update,
    )

    h_V = torch.randn(B, L, C, requires_grad=True)
    h_E = torch.randn(B, L, 8, C)  # not consumed by ContextModule but part of signature
    V = torch.randn(B, L, M, C)
    Y_nodes = torch.randn(B, L, M, C)
    Y_edges = torch.randn(B, L, M, M, C)
    Y_m = torch.ones(B, L, M)
    prot_mask = torch.ones(B, L)

    out = module(
        h_V=h_V,
        h_E=h_E,
        V=V,
        Y_nodes=Y_nodes,
        Y_edges=Y_edges,
        Y_m=Y_m,
        protein_residue_node_mask=prot_mask,
    )

    assert out.shape == h_V.shape, f"ContextModule output shape {tuple(out.shape)} != input {tuple(h_V.shape)}"
    assert not torch.isnan(out).any(), "ContextModule output contains NaN"

    loss = out.sum()
    loss.backward()

    # V_C / V_C_norm must always receive gradient (residual channel to the protein track).
    _check_grad_for_keywords(module, ["V_C"])

    if context_feature_edge_update:
        # W11/W12/W13/norm3 are exercised only when edge update is ON.
        _check_grad_for_keywords(
            module,
            [
                "context_feature_processor.0.W11", "context_feature_processor.0.W13",
                "context_feature_aggregator.0.W11", "context_feature_aggregator.0.W13",
            ],
        )
    print(f"[OK] ContextModule context_feature_edge_update={context_feature_edge_update} out.shape={tuple(out.shape)}")


def test_enclayer(is_last_layer: bool) -> None:
    torch.manual_seed(0)
    B, L, K, C = 2, 6, 4, 32

    # EncLayer is constructed with num_in = 2*C (since the message concatenates
    # h_V_expand + cat_neighbors_nodes(h_V, h_E) = h_V_i + h_V_j + h_E).
    # The h_E input itself lives in hidden_dim (after W_e projection in AtomMPNN).
    layer = EncLayer(C, C * 2, dropout=0.1, is_last_layer=is_last_layer)
    layer.eval()

    h_V = torch.randn(B, L, C, requires_grad=True)
    h_E = torch.randn(B, L, K, C, requires_grad=True)
    E_idx = torch.randint(0, L, (B, L, K))

    out_V, out_E = layer(h_V, h_E, E_idx)

    assert out_V.shape == (B, L, C)
    assert out_E.shape == (B, L, K, C)

    if is_last_layer:
        assert torch.equal(out_E, h_E), "is_last_layer=True should leave h_E unchanged"
    else:
        assert not torch.equal(out_E, h_E), "is_last_layer=False should modify h_E"

    (out_V.sum() + out_E.sum()).backward()
    if not is_last_layer:
        _check_grad_for_keywords(layer, ["W11", "W12", "W13", "norm3"])

    print(f"[OK] EncLayer is_last_layer={is_last_layer} out_V.shape={tuple(out_V.shape)} out_E.shape={tuple(out_E.shape)}")


def main() -> None:
    test_context_module(context_feature_edge_update=False)
    test_context_module(context_feature_edge_update=True)
    test_enclayer(is_last_layer=False)
    test_enclayer(is_last_layer=True)
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
