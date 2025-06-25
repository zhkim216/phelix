import unittest
import torch
from torchtyping import TensorType
from typing import Any, Dict
from allatom_design.model.seq_denoiser.denoisers.atom_mpnn_denoiser import _aggregate_potts_params


class TestPottsAggregation(unittest.TestCase):
    def setUp(self):
        """Set up a dummy model instance for testing."""
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def test_simple_aggregation(self):
        """Test aggregation of two inputs into a single group."""
        B, N, C = 2, 3, 2
        
        h = torch.randn(B, N, C, device=self.device, dtype=self.dtype)
        J = torch.randn(B, N, N, C, C, device=self.device, dtype=self.dtype)
        
        potts_decoder_aux = {
            "h": h,
            "J": J,
            "edge_idx": torch.arange(N, device=self.device).expand(B, N, N),
            "mask_i": torch.ones(B, N, device=self.device, dtype=self.dtype),
            "mask_ij": torch.ones(B, N, N, device=self.device, dtype=self.dtype),
        }
        
        tied_sampling_inputs = {
            "inverse": torch.tensor([0, 0], device=self.device),
            "unique_ids": torch.tensor([0], device=self.device),
        }
        
        result = _aggregate_potts_params(potts_decoder_aux, tied_sampling_inputs)
        
        # Expected outputs
        expected_h = h[0] + h[1]
        expected_J = J[0] + J[1]
        expected_mask_i = torch.ones(1, N, device=self.device, dtype=self.dtype)
        expected_mask_ij = torch.ones(1, N, N, device=self.device, dtype=self.dtype)
        
        torch.testing.assert_close(result["h"], expected_h.unsqueeze(0))
        torch.testing.assert_close(result["J"], expected_J.unsqueeze(0))
        torch.testing.assert_close(result["mask_i"], expected_mask_i)
        torch.testing.assert_close(result["mask_ij"], expected_mask_ij)
        self.assertEqual(result["h"].shape[0], 1) # Should be one group

    def test_partial_masking(self):
        """Test aggregation with partial node and edge masks."""
        B, N, C = 2, 3, 2
        
        h = torch.ones(B, N, C, device=self.device, dtype=self.dtype)
        J = torch.ones(B, N, N, C, C, device=self.device, dtype=self.dtype)
        
        mask_i = torch.tensor([
            [1., 1., 1.],
            [1., 1., 0.]  # Node 2 is masked in the second input
        ], device=self.device, dtype=self.dtype)
        
        mask_ij = torch.tensor([
            [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
            [[1., 1., 0.], [1., 1., 0.], [0., 0., 0.]] # Edges to/from node 2 are masked
        ], device=self.device, dtype=self.dtype)

        potts_decoder_aux = {
            "h": h,
            "J": J * mask_ij.unsqueeze(-1).unsqueeze(-1),
            "edge_idx": torch.arange(N, device=self.device).expand(B, N, N),
            "mask_i": mask_i,
            "mask_ij": mask_ij,
        }
        
        tied_sampling_inputs = {
            "inverse": torch.tensor([0, 0], device=self.device),
            "unique_ids": torch.tensor([0], device=self.device),
        }
        
        result = _aggregate_potts_params(potts_decoder_aux, tied_sampling_inputs)

        # Expected h: Node 2 is masked, so its h should be the sum but the mask will be 0.
        expected_h = h[0] + h[1] 
        # Expected mask_i: Only nodes 0 and 1 are present in both.
        expected_mask_i = torch.tensor([[1., 1., 0.]], device=self.device, dtype=self.dtype)

        # Expected J: Sum of Js where edges exist.
        expected_J = potts_decoder_aux["J"][0] + potts_decoder_aux["J"][1]
        
        # Expected mask_ij: An edge exists if it exists in at least one input AND both its nodes are unmasked.
        # Edge counts will be > 0 for all but connections to node 2 from the second input.
        # But mask_i_new will zero out all connections to node 2.
        expected_mask_ij = torch.tensor([[
            [1., 1., 0.],
            [1., 1., 0.],
            [0., 0., 0.]
        ]], device=self.device, dtype=self.dtype)
        
        torch.testing.assert_close(result["h"], expected_h.unsqueeze(0))
        torch.testing.assert_close(result["J"], expected_J.unsqueeze(0))
        torch.testing.assert_close(result["mask_i"], expected_mask_i)
        torch.testing.assert_close(result["mask_ij"], expected_mask_ij)

    def test_multiple_groups(self):
        """Test aggregation with multiple distinct groups in one batch."""
        B, N, C = 4, 2, 2
        
        h = torch.randn(B, N, C, device=self.device, dtype=self.dtype)
        J = torch.randn(B, N, N, C, C, device=self.device, dtype=self.dtype)
        
        potts_decoder_aux = {
            "h": h,
            "J": J,
            "edge_idx": torch.arange(N, device=self.device).expand(B, N, N),
            "mask_i": torch.ones(B, N, device=self.device, dtype=self.dtype),
            "mask_ij": torch.ones(B, N, N, device=self.device, dtype=self.dtype),
        }
        
        tied_sampling_inputs = {
            "inverse": torch.tensor([0, 1, 0, 1], device=self.device), # items 0,2 -> grp 0; 1,3 -> grp 1
            "unique_ids": torch.tensor([0, 1], device=self.device),
        }
        
        result = _aggregate_potts_params(potts_decoder_aux, tied_sampling_inputs)
        
        # Expected outputs
        expected_h_grp0 = h[0] + h[2]
        expected_h_grp1 = h[1] + h[3]
        expected_h = torch.stack([expected_h_grp0, expected_h_grp1])
        
        expected_J_grp0 = J[0] + J[2]
        expected_J_grp1 = J[1] + J[3]
        expected_J = torch.stack([expected_J_grp0, expected_J_grp1])
        
        expected_mask_i = torch.ones(2, N, device=self.device, dtype=self.dtype)
        expected_mask_ij = torch.ones(2, N, N, device=self.device, dtype=self.dtype)
        
        torch.testing.assert_close(result["h"], expected_h)
        torch.testing.assert_close(result["J"], expected_J)
        torch.testing.assert_close(result["mask_i"], expected_mask_i)
        torch.testing.assert_close(result["mask_ij"], expected_mask_ij)
        self.assertEqual(result["h"].shape[0], 2) # Should be two groups


    def test_multiple_groups(self):
        """Test aggregation with multiple distinct groups in one batch."""
        B, N, C = 4, 2, 2
        
        h = torch.randn(B, N, C, device=self.device, dtype=self.dtype)
        J = torch.randn(B, N, N, C, C, device=self.device, dtype=self.dtype)
        
        potts_decoder_aux = {
            "h": h,
            "J": J,
            "edge_idx": torch.arange(N, device=self.device).expand(B, N, N),
            "mask_i": torch.ones(B, N, device=self.device, dtype=self.dtype),
            "mask_ij": torch.ones(B, N, N, device=self.device, dtype=self.dtype),
        }
        
        tied_sampling_inputs = {
            "inverse": torch.tensor([0, 1, 0, 1], device=self.device), # items 0,2 -> grp 0; 1,3 -> grp 1
            "unique_ids": torch.tensor([0, 1], device=self.device),
        }
        
        result = _aggregate_potts_params(potts_decoder_aux, tied_sampling_inputs)
        
        # Expected outputs
        expected_h_grp0 = h[0] + h[2]
        expected_h_grp1 = h[1] + h[3]
        expected_h = torch.stack([expected_h_grp0, expected_h_grp1])
        
        expected_J_grp0 = J[0] + J[2]
        expected_J_grp1 = J[1] + J[3]
        expected_J = torch.stack([expected_J_grp0, expected_J_grp1])
        
        expected_mask_i = torch.ones(2, N, device=self.device, dtype=self.dtype)
        expected_mask_ij = torch.ones(2, N, N, device=self.device, dtype=self.dtype)
        
        torch.testing.assert_close(result["h"], expected_h)
        torch.testing.assert_close(result["J"], expected_J)
        torch.testing.assert_close(result["mask_i"], expected_mask_i)
        torch.testing.assert_close(result["mask_ij"], expected_mask_ij)
        self.assertEqual(result["h"].shape[0], 2) # Should be two groups



    def test_sparse_and_reordered_edges(self):
        """Test aggregation with sparse (N,K) and reordered edges including self-loops."""
        B, N, K, C = 3, 3, 2, 2
        
        # Each graph has edges that include a self-loop for each node.
        edge_idx = torch.tensor([
            [[0, 1], [1, 2], [2, 0]], # G0: 0->(0,1), 1->(1,2), 2->(2,0)
            [[0, 2], [1, 0], [2, 1]], # G1: 0->(0,2), 1->(1,0), 2->(2,1)
            [[1, 0], [2, 1], [0, 2]], # G2: 0->(1,0), 1->(2,1), 2->(0,2) (reordered)
        ], device=self.device)

        j_vals = torch.arange(1, B * N * K + 1, device=self.device, dtype=self.dtype).view(B, N, K)
        J = j_vals.unsqueeze(-1).unsqueeze(-1) * torch.eye(C, device=self.device, dtype=self.dtype)

        # Mask the self-loop for node 1 in graph 2 for testing.
        mask_ij = torch.ones(B, N, K, device=self.device, dtype=self.dtype)
        mask_ij[2, 1, 1] = 0 # In G2, for node 1, the 2nd neighbor is 1 (self-loop). Mask it.

        potts_decoder_aux = {
            "h": torch.ones(B, N, C, device=self.device, dtype=self.dtype),
            "J": J * mask_ij.unsqueeze(-1).unsqueeze(-1),
            "edge_idx": edge_idx,
            "mask_i": torch.ones(B, N, device=self.device, dtype=self.dtype),
            "mask_ij": mask_ij,
        }
        
        tied_sampling_inputs = {
            "inverse": torch.tensor([0, 0, 0], device=self.device),
            "unique_ids": torch.tensor([0], device=self.device),
        }

        result = _aggregate_potts_params(potts_decoder_aux, tied_sampling_inputs)

        # Hardcode the expected J_new by summing contributions for each edge (i,j)
        expected_J = torch.zeros(1, N, N, C, C, device=self.device, dtype=self.dtype)
        J_sum_source = potts_decoder_aux["J"]
        
        # Self-loops
        expected_J[0, 0, 0] = J_sum_source[0, 0, 0] + J_sum_source[1, 0, 0] + J_sum_source[2, 0, 1]
        expected_J[0, 1, 1] = J_sum_source[0, 1, 0] + J_sum_source[1, 1, 0] # G2 self-loop is masked
        expected_J[0, 2, 2] = J_sum_source[0, 2, 0] + J_sum_source[1, 2, 0] + J_sum_source[2, 2, 1]

        # Other edges
        expected_J[0, 0, 1] = J_sum_source[0, 0, 1] + J_sum_source[2, 0, 0]
        expected_J[0, 0, 2] = J_sum_source[1, 0, 1]
        expected_J[0, 1, 0] = J_sum_source[1, 1, 1]
        expected_J[0, 1, 2] = J_sum_source[0, 1, 1] + J_sum_source[2, 1, 0]
        expected_J[0, 2, 0] = J_sum_source[0, 2, 1] + J_sum_source[2, 2, 0]
        expected_J[0, 2, 1] = J_sum_source[1, 2, 1]
        
        torch.testing.assert_close(result["J"], expected_J)
        
        # Since every possible edge (including self-loops) exists in at least one graph,
        # the final mask should be all ones.
        expected_mask_ij = torch.ones(1, N, N, device=self.device, dtype=self.dtype)
        torch.testing.assert_close(result["mask_ij"], expected_mask_ij)


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
