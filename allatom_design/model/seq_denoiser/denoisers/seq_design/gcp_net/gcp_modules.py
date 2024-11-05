# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------
from copy import copy
from functools import partial
from typing import Any, Optional, Tuple, Union
from omegaconf import DictConfig
from torchtyping import TensorType 
from typeguard import typechecked
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from allatom_design.model.seq_denoiser.denoisers.seq_design.gcp_net.gcp_utils import (
    GCPLayerNorm, GCPDropout, ScalarVector, 
    is_identity, safe_norm, scalarize, vectorize, get_nonlinearity)

from allatom_design.data import residue_constants as rc 

from allatom_design.data.data import (
    get_rc_tensor,  
    orientations,
    dihedrals,
    sidechains,
    positional_embeddings,
    normalize,
    rbf,
    nan_to_num,
    dist
)

class GCP2(nn.Module):
    def __init__(
            self,
            input_dims: ScalarVector,
            output_dims: ScalarVector,
            cfg: DictConfig
    ):
        super(GCP2, self).__init__()
        nonlinearities = (None, None) if cfg.nonlinearities is None else cfg.nonlinearities
        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.scalar_nonlinearity, self.vector_nonlinearity = (
            get_nonlinearity(nonlinearities[0], return_functional=True),
            get_nonlinearity(nonlinearities[1], return_functional=True)
        )
        self.scalar_gate, self.vector_gate, self.frame_gate, self.sigma_frame_gate = (
            cfg.scalar_gate, cfg.vector_gate, cfg.frame_gate, cfg.sigma_frame_gate
        )
        self.vector_residual, self.vector_frame_residual = cfg.vector_residual, cfg.vector_frame_residual
        self.ablate_frame_updates = cfg.ablate_frame_updates
        self.ablate_scalars, self.ablate_vectors = cfg.ablate_scalars, cfg.ablate_vectors
        self.enable_e3_equivariance = cfg.enable_e3_equivariance

        if self.scalar_gate > 0:
            self.norm = nn.LayerNorm(self.scalar_output_dim)

        if self.vector_input_dim:
            assert (
                self.vector_input_dim % cfg.bottleneck == 0
            ), f"Input channel of vector ({self.vector_input_dim}) must be divisible with bottleneck factor ({cfg.bottleneck})"

            self.hidden_dim = self.vector_input_dim // cfg.bottleneck if cfg.bottleneck > 1 else max(self.vector_input_dim,
                                                                                             self.vector_output_dim)

            scalar_vector_frame_dim = (cfg.scalarization_vectorization_output_dim *
                                       3) if not self.ablate_frame_updates else 0
            self.vector_down = nn.Linear(self.vector_input_dim, self.hidden_dim, bias=False)
            self.scalar_out = nn.Linear(self.hidden_dim + self.scalar_input_dim +
                                        scalar_vector_frame_dim, self.scalar_output_dim)

            if not self.ablate_frame_updates:
                self.vector_down_frames = nn.Linear(
                    self.vector_input_dim, cfg.scalarization_vectorization_output_dim, bias=False)

            if self.vector_output_dim:
                self.vector_up = nn.Linear(self.hidden_dim, self.vector_output_dim, bias=False)
                if not self.ablate_frame_updates:
                    if self.frame_gate:
                        self.vector_out_scale_frames = nn.Linear(
                            self.scalar_output_dim, cfg.scalarization_vectorization_output_dim * 3)
                        self.vector_up_frames = nn.Linear(
                            cfg.scalarization_vectorization_output_dim, self.vector_output_dim, bias=False)
                    elif self.vector_gate:
                        self.vector_out_scale = nn.Linear(self.scalar_output_dim, self.vector_output_dim)
                elif self.vector_gate:
                    self.vector_out_scale = nn.Linear(self.scalar_output_dim, self.vector_output_dim)
        else:
            self.scalar_out = nn.Linear(self.scalar_input_dim, self.scalar_output_dim)

    @typechecked
    def create_zero_vector(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"]
    ) -> TensorType["batch_num_entities", "o", 3]:
        return torch.zeros(scalar_rep.shape[0], self.vector_output_dim, 3, device=scalar_rep.device)

    @typechecked
    def process_vector_without_frames(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"],
        v_pre: TensorType["batch_num_entities", 3, "m"],
        vector_hidden_rep: TensorType["batch_num_entities", 3, "n"]
    ) -> TensorType["batch_num_entities", "o", 3]:
        vector_rep = self.vector_up(vector_hidden_rep)
        if self.vector_residual:
            vector_rep = vector_rep + v_pre
        vector_rep = vector_rep.transpose(-1, -2)

        if self.vector_gate:
            gate = self.vector_out_scale(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(vector_rep, dim=-1, keepdim=True))

        return vector_rep

    @typechecked
    def process_vector_with_frames(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"],
        v_pre: TensorType["batch_num_entities", 3, "m"],
        vector_hidden_rep: TensorType["batch_num_entities", 3, "n"],
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_inputs: bool,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> TensorType["batch_num_entities", "o", 3]:
        vector_rep = self.vector_up(vector_hidden_rep)
        if self.vector_residual:
            vector_rep = vector_rep + v_pre
        vector_rep = vector_rep.transpose(-1, -2)

        if self.frame_gate:
            # derive vector features from direction-robust frames
            gate = self.vector_out_scale_frames(self.vector_nonlinearity(scalar_rep))
            # perform frame-gating, where edges must be present
            gate_vector = vectorize(
                gate,
                edge_index,
                frames,
                node_inputs=node_inputs,
                dim_size=scalar_rep.shape[0],
                node_mask=node_mask
            )
            # ensure frame vector channels for `coordinates` are being left-multiplied
            gate_vector_rep = self.vector_up_frames(gate_vector.transpose(-1, -2)).transpose(-1, -2)
            # apply row-wise scalar gating with frame vector
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(gate_vector_rep, dim=-1, keepdim=True))
        elif self.vector_gate:
            gate = self.vector_out_scale(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(vector_rep, dim=-1, keepdim=True))

        return vector_rep

    @typechecked
    def forward(
        self,
        s_maybe_v: Union[
            Tuple[
                TensorType["batch_num_entities", "scalar_dim"],
                TensorType["batch_num_entities", "m", "vector_dim"]
            ],
            TensorType["batch_num_entities", "merged_scalar_dim"]
        ],
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_inputs: bool = False,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> Union[
        Tuple[
            TensorType["batch_num_entities", "new_scalar_dim"],
            TensorType["batch_num_entities", "n", "vector_dim"]
        ],
        TensorType["batch_num_entities", "new_scalar_dim"]
    ]:
        if self.vector_input_dim:
            scalar_rep, vector_rep = s_maybe_v
            scalar_rep = torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
            vector_rep = torch.zeros_like(vector_rep) if self.ablate_vectors else vector_rep
            v_pre = vector_rep.transpose(-1, -2)

            vector_hidden_rep = self.vector_down(v_pre)
            vector_norm = safe_norm(vector_hidden_rep, dim=-2)
            merged = torch.cat((scalar_rep, vector_norm), dim=-1)

            if not self.ablate_frame_updates:
                # GCP2: curate direction-robust scalar geometric features
                vector_down_frames_hidden_rep = self.vector_down_frames(v_pre)
                scalar_hidden_rep = scalarize(
                    vector_down_frames_hidden_rep.transpose(-1, -2),
                    edge_index,
                    frames,
                    node_inputs=node_inputs,
                    enable_e3_equivariance=self.enable_e3_equivariance,
                    dim_size=vector_down_frames_hidden_rep.shape[0],
                    node_mask=node_mask
                )
                merged = torch.cat((merged, scalar_hidden_rep), dim=-1)
        else:
            # bypass updating scalar features using vector information
            merged = s_maybe_v

        scalar_rep = self.scalar_out(merged)

        if not self.vector_output_dim:
            # bypass updating vector features using scalar information
            scalar_rep = torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
            return self.scalar_nonlinearity(scalar_rep)
        elif self.vector_output_dim and not self.vector_input_dim:
            # instantiate vector features that are learnable in proceeding GCP layers
            vector_rep = self.create_zero_vector(scalar_rep)
        elif self.ablate_frame_updates:
            # GCP-Baseline: update vector features using row-wise scalar gating
            vector_rep = self.process_vector_without_frames(scalar_rep, v_pre, vector_hidden_rep)
        else:
            # GCP2: update vector features using either row-wise scalar gating with complete local frames or row-wise self-scalar gating
            vector_rep = self.process_vector_with_frames(
                scalar_rep,
                v_pre,
                vector_hidden_rep,
                edge_index,
                frames,
                node_inputs=node_inputs,
                node_mask=node_mask
            )

        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        scalar_rep = torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
        vector_rep = torch.zeros_like(vector_rep) if self.ablate_vectors else vector_rep
        return ScalarVector(scalar_rep, vector_rep)

class GCPEmbedding(nn.Module):
    def __init__(
        self,
        edge_input_dims: ScalarVector,
        node_input_dims: ScalarVector,
        edge_hidden_dims: ScalarVector,
        node_hidden_dims: ScalarVector,
        num_atom_types: int = rc.atom_type_num,
        cfg: DictConfig = None,
        pre_norm: bool = True
    ):
        super(GCPEmbedding, self).__init__()

        self.pre_norm = pre_norm
        if pre_norm:
            self.edge_normalization = GCPLayerNorm(edge_input_dims)
            self.node_normalization = GCPLayerNorm(node_input_dims)
        else:
            self.edge_normalization = GCPLayerNorm(edge_hidden_dims)
            self.node_normalization = GCPLayerNorm(node_hidden_dims)

        self.edge_embedding = GCP2(
            edge_input_dims,
            edge_hidden_dims,
            cfg,
        )

        self.node_embedding = GCP2(
            node_input_dims,
            node_hidden_dims,
            cfg,
        )

    @typechecked
    def forward(
        self,
        batch: Batch
    ) -> Tuple[
        Union[
            Tuple[
                TensorType["batch_num_nodes", "h_hidden_dim"],
                TensorType["batch_num_nodes", "m", "chi_hidden_dim"]
            ],
            TensorType["batch_num_nodes", "h_hidden_dim"]
        ],
        Union[
            Tuple[
                TensorType["batch_num_edges", "e_hidden_dim"],
                TensorType["batch_num_edges", "x", "xi_hidden_dim"]
            ],
            TensorType["batch_num_edges", "e_hidden_dim"]
        ]
    ]:
        node_rep = ScalarVector(batch.h, batch.chi)
        edge_rep = ScalarVector(batch.e, batch.xi)

        edge_rep = edge_rep.scalar if not self.edge_embedding.vector_input_dim else edge_rep
        node_rep = node_rep.scalar if not self.node_embedding.vector_input_dim else node_rep

        if self.pre_norm:
            edge_rep = self.edge_normalization(edge_rep)
            node_rep = self.node_normalization(node_rep)

        edge_rep = self.edge_embedding(
            edge_rep,
            batch.edge_index,
            batch.f_ij,
            node_inputs=False,
            node_mask=getattr(batch, "mask", None)
        )

        node_rep = self.node_embedding(
            node_rep,
            batch.edge_index,
            batch.f_ij,
            node_inputs=True,
            node_mask=getattr(batch, "mask", None)
        )

        if not self.pre_norm:
            edge_rep = self.edge_normalization(edge_rep)
            node_rep = self.node_normalization(node_rep)

        return node_rep, edge_rep

class GCPMessagePassing(nn.Module):
    def __init__(
        self,
        input_dims: ScalarVector,
        output_dims: ScalarVector,
        edge_dims: ScalarVector,
        cfg: DictConfig,
        mp_cfg: DictConfig,
        reduce_function: str = "mean",
        use_scalar_message_attention: bool = False,
        aggregate_with_row: bool = False
    ):
        super().__init__()

        # hyperparameters
        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.edge_scalar_dim, self.edge_vector_dim = edge_dims
        self.conv_cfg = mp_cfg
        self.self_message = self.conv_cfg.self_message
        self.use_residual_message_gcp = self.conv_cfg.use_residual_message_gcp
        self.reduce_function = reduce_function
        self.use_scalar_message_attention = use_scalar_message_attention
        self.aggregate_with_row = aggregate_with_row

        scalars_in_dim = 2 * self.scalar_input_dim + self.edge_scalar_dim
        vectors_in_dim = 2 * self.vector_input_dim + self.edge_vector_dim

        # config instantiations
        soft_cfg = copy(cfg)
        soft_cfg.bottleneck, soft_cfg.vector_residual = cfg.default_bottleneck, cfg.default_vector_residual
        soft_cfg.nonlinearities = cfg.nonlinearities if self.conv_cfg.num_message_layers > 1 else None
        primary_cfg_GCP = partial(GCP2, cfg=soft_cfg)
        secondary_cfg_GCP = partial(GCP2, cfg=cfg)

        # PyTorch modules #
        module_list = [
            primary_cfg_GCP(
                (scalars_in_dim, vectors_in_dim),
                output_dims
            )
        ]

        for _ in range(self.conv_cfg.num_message_layers - 2):
            module_list.append(secondary_cfg_GCP(output_dims, output_dims))

        if self.conv_cfg.num_message_layers > 1:
            soft_cfg.nonlinearities=(None, None)
            module_list.append(primary_cfg_GCP(output_dims, output_dims))

        self.message_fusion = nn.ModuleList(module_list)

        # learnable scalar message gating
        if use_scalar_message_attention:
            self.scalar_message_attention = nn.Sequential(
                nn.Linear(output_dims.scalar, 1),
                nn.Sigmoid()
            )

    @typechecked
    def message(
        self,
        node_rep: ScalarVector,
        edge_rep: ScalarVector,
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> TensorType["batch_num_edges", "message_dim"]:
        row, col = edge_index
        vector = node_rep.vector.reshape(node_rep.vector.shape[0], node_rep.vector.shape[1] * node_rep.vector.shape[2])
        vector_reshaped = ScalarVector(node_rep.scalar, vector)

        s_row, v_row = vector_reshaped.idx(row)
        s_col, v_col = vector_reshaped.idx(col)

        v_row = v_row.reshape(v_row.shape[0], v_row.shape[1] // 3, 3)
        v_col = v_col.reshape(v_col.shape[0], v_col.shape[1] // 3, 3)

        message = ScalarVector(s_row, v_row).concat((edge_rep, ScalarVector(s_col, v_col)))

        if self.use_residual_message_gcp:
            message_residual = self.message_fusion[0](message, edge_index, frames, node_inputs=False, node_mask=node_mask)
            for module in self.message_fusion[1:]:
                # ResGCP: exchange geometric messages while maintaining residual connection to original message
                new_message = module(message_residual, edge_index, frames, node_inputs=False, node_mask=node_mask)
                message_residual = message_residual + new_message
        else:
            message_residual = message
            for module in self.message_fusion:
                # ablate ResGCP: exchange geometric messages without maintaining residual connection to original message
                message_residual = module(message_residual, edge_index, frames, node_inputs=False, node_mask=node_mask)

        # learn to gate scalar messages
        if self.use_scalar_message_attention:
            message_residual_attn = self.scalar_message_attention(message_residual.scalar)
            message_residual = ScalarVector(message_residual.scalar * message_residual_attn, message_residual.vector)

        return message_residual.flatten()

    @typechecked
    def aggregate(
        self,
        message: TensorType["batch_num_edges", "message_dim"],
        edge_index: TensorType[2, "batch_num_edges"],
        dim_size: int
    ) -> TensorType["batch_num_nodes", "aggregate_dim"]:
        row, col = edge_index
        index = row if self.aggregate_with_row else col

        # Initialize an output tensor with the same shape as the required `dim_size`
        output_scatter_reduce = torch.zeros((dim_size, message.shape[1]), dtype=message.dtype, device=message.device)
        
        #the -1 index is ignored in torch_scatter.scatter, but for scatter_reduce we have to explicitly remove it
        valid_mask = index != -1
        index_filtered = index[valid_mask]
        message_filtered = message[valid_mask]        

        aggregate = output_scatter_reduce.scatter_reduce(
            dim=0,                        
            index=index_filtered.unsqueeze(-1).expand_as(message_filtered),                  
            src=message_filtered,                 
            reduce=self.reduce_function,  
            include_self = False,
        )        
        return aggregate

    @typechecked
    def forward(
        self,
        node_rep: ScalarVector,
        edge_rep: ScalarVector,
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> ScalarVector:
        message = self.message(node_rep, edge_rep, edge_index, frames, node_mask=node_mask)
        aggregate = self.aggregate(message, edge_index, dim_size=node_rep.scalar.shape[0])
        return ScalarVector.recover(aggregate, self.vector_output_dim)
    
class GCPInteractions(nn.Module):
    def __init__(
        self,
        node_dims: ScalarVector,
        edge_dims: ScalarVector,
        cfg: DictConfig,
        layer_cfg: DictConfig,
        dropout: float = 0.1,
        autoregressive: bool = False,
        nonlinearities: Optional[Tuple[Any, Any]] = None,
        updating_node_positions: bool = False
    ):
        super().__init__()

        # hyperparameters #
        if nonlinearities is None:
            nonlinearities = cfg.nonlinearities
        self.pre_norm = layer_cfg.pre_norm
        self.updating_node_positions = updating_node_positions
        self.ablate_x_force_update = getattr(cfg, "ablate_x_force_update", True)
        self.node_positions_weight = getattr(cfg, "node_positions_weight", 1.0)
        reduce_function = "add" if autoregressive else "mean"

        # PyTorch modules #

        # geometry-complete message-passing neural network
        message_function = GCPMessagePassing

        self.interaction = message_function(
            node_dims,
            node_dims,
            edge_dims,
            reduce_function=reduce_function,
            cfg=cfg,
            mp_cfg=layer_cfg.mp
        )

        # config instantiations
        ff_cfg = copy(cfg)
        ff_cfg.nonlinearities = nonlinearities
        ff_without_res_cfg = copy(cfg)
        ff_without_res_cfg.vector_residual = False

        ff_GCP = partial(GCP2, cfg=ff_cfg)
        ff_without_res_GCP = partial(GCP2, cfg=ff_without_res_cfg)

        self.gcp_norm = nn.ModuleList([GCPLayerNorm(node_dims) for _ in range(2)])
        self.gcp_dropout = nn.ModuleList([GCPDropout(dropout) for _ in range(2)])

        # build out feedforward (FF) network modules
        ff_interaction_layers = []
        hidden_dims = node_dims if layer_cfg.num_feedforward_layers == 1 else 4 * node_dims.scalar, 2 * node_dims.vector
        ff_interaction_layers.append(
            ff_without_res_GCP(
                node_dims, hidden_dims
            )
        )

        interaction_layers = [
            ff_GCP(hidden_dims, hidden_dims, enable_e3_equivariance=cfg.enable_e3_equivariance)
            for _ in range(layer_cfg.num_feedforward_layers - 2)
        ]
        ff_interaction_layers.extend(interaction_layers)

        if layer_cfg.num_feedforward_layers > 1:
            ff_without_res_cfg.nonlinearities=(None, None)
            ff_interaction_layers.append(
                ff_without_res_GCP(
                    hidden_dims, node_dims,
                )
            )

        self.feedforward_network = nn.ModuleList(ff_interaction_layers)

        # potentially build out node position update modules
        if updating_node_positions:
            # node position update GCPs
            node_position_update_gcps = [
                ff_without_res_GCP(
                    node_dims, (node_dims.scalar, 1),
                    nonlinearities=cfg.nonlinearities,
                    enable_e3_equivariance=cfg.enable_e3_equivariance
                )
            ]
            self.node_position_update_network = nn.ModuleList(node_position_update_gcps)

            # node position force-update layers
            scalar_hidden_dim = node_dims.scalar
            scalar_nonlinearity = cfg.nonlinearities[0]
            self.phi_force_i = None if self.ablate_x_force_update else nn.Linear(scalar_hidden_dim, scalar_hidden_dim)
            self.phi_force_j = None if self.ablate_x_force_update else nn.Linear(scalar_hidden_dim, scalar_hidden_dim)
            phi_x_force_ij_layer = None if self.ablate_x_force_update else nn.Linear(scalar_hidden_dim, 3, bias=False)
            None if self.ablate_x_force_update else torch.nn.init.xavier_uniform_(
                phi_x_force_ij_layer.weight, gain=0.001)
            self.phi_force_ij = None if self.ablate_x_force_update else nn.Sequential(
                get_nonlinearity(scalar_nonlinearity, layer_cfg.nonlinearity_slope),
                phi_x_force_ij_layer
            )

    @typechecked
    def derive_x_update(
        self,
        node_rep: ScalarVector,
        edge_index: TensorType[2, "batch_num_edges"],
        f_ij: TensorType["batch_num_edges", 3, 3],
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> TensorType["batch_num_nodes", 3]:
        row, col = edge_index

        # VectorUpdate: use vector-valued features to derive node position updates
        (h_v, chi_v) = node_rep
        for position_update_gcp in self.node_position_update_network:
            (h_v, chi_v) = position_update_gcp(
                (h_v, chi_v),
                edge_index,
                f_ij,
                node_inputs=True,
                node_mask=node_mask
            )

        # ForceUpdate: use inter-atom forces to derive node position updates from each neighboring node
        if self.ablate_x_force_update:
            x_force_update = torch.zeros((h_v.shape[0], 3), device=h_v.device)
        else:
            f_ij = f_ij.reshape(f_ij.shape[0], 1, -1)
            x_diff, x_cross, x_vertical = f_ij[:, :, :3].squeeze(), f_ij[:, :, 3:6].squeeze(), f_ij[:, :, 6:].squeeze()
            
            h_i, h_j = h_v[row], h_v[col]
            x_force_coef = self.phi_force_ij(self.phi_force_i(h_i) + self.phi_force_j(h_j))
            x_force_update = (
                x_force_coef[:, :1] * x_diff + x_force_coef[:, 1:2] * x_cross + x_force_coef[:, 2:3] * x_vertical
            )

            # summarize node position updates across neighboring nodes
            dim_size = node_rep[0].shape[0]
            x_force_update_aggregate = torch.zeros(dim_size, dtype=x_force_update.dtype, device=x_force_update.device)

            # Apply scatter_reduce to perform mean reduction
            x_force_update = x_force_update_aggregate.scatter_reduce_(
                dim=0,                   # Dimension to scatter along
                index=col,               # Index tensor
                src=x_force_update,      # Source tensor
                reduce="mean"            # Reduction operation
            )
        # combine scalar and vector-valued features to curate a single positional update for each node
        x_update = (chi_v.squeeze(1) + x_force_update) * self.node_positions_weight  # (up/down)weight position updates

        return x_update.clamp(min=-100, max=100)  # note: not used but may save training

    @typechecked
    def forward(
        self,
        node_rep: Tuple[TensorType["batch_num_nodes", "node_hidden_dim"], TensorType["batch_num_nodes", "m", 3]],
        edge_rep: Tuple[TensorType["batch_num_edges", "edge_hidden_dim"], TensorType["batch_num_edges", "x", 3]],
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        node_pos: Optional[TensorType["batch_num_nodes", 3]] = None
    ) -> Union[
        Tuple[
            TensorType["batch_num_nodes", "hidden_dim"],
            TensorType["batch_num_nodes", "n", 3]
        ],
        Tuple[
            Tuple[
                TensorType["batch_num_nodes", "hidden_dim"],
                TensorType["batch_num_nodes", "n", 3]
            ],
            TensorType["batch_num_nodes", 3]
        ]
    ]:
        node_rep = ScalarVector(node_rep[0], node_rep[1])
        edge_rep = ScalarVector(edge_rep[0], edge_rep[1])

        # apply GCP normalization (1)
        if self.pre_norm:
            node_rep = self.gcp_norm[0](node_rep)
        
        hidden_residual = self.interaction(
                node_rep, edge_rep, edge_index, frames, node_mask=node_mask
            )
    
        # apply GCP dropout (1)
        node_rep = node_rep + self.gcp_dropout[0](hidden_residual)

        # apply GCP normalization (2)
        if self.pre_norm:
            node_rep = self.gcp_norm[1](node_rep)
        else:
            node_rep = self.gcp_norm[0](node_rep)

        # propagate with feedforward layers
        hidden_residual = node_rep
        for module in self.feedforward_network:
            hidden_residual = module(
                hidden_residual,
                edge_index,
                frames,
                node_inputs=True,
                node_mask=node_mask
            )

        # apply GCP dropout (2)
        node_rep = node_rep + self.gcp_dropout[1](hidden_residual)

        # apply GCP normalization (3)
        if not self.pre_norm:
            node_rep = self.gcp_norm[1](node_rep)

        # bypass updating node positions
        if not self.updating_node_positions:
            return node_rep

        # update node positions
        node_pos = node_pos + self.derive_x_update(
            node_rep, edge_index, frames, node_mask=node_mask
        )

        return node_rep, node_pos

class GCPInputFeaturizer(nn.Module):
    def __init__(self,
                num_positional_embeddings
                ):
        
        super().__init__()
        self.num_positional_embeddings = num_positional_embeddings
        self.zero_ghost_atoms = False
        self.batch = Batch.from_data_list([Data()])
    
    def forward(self, coords, seq, mpnn_E_idx, padding_mask, atom14_mask):
        node_features = self.get_node_features(coords, padding_mask, atom14_mask)
        edge_features, edge_index = self.get_edge_features(
            coords, padding_mask, mpnn_E_idx, atom14_mask)
        atom_type = self.get_atom_type(seq, atom14_mask)
        
        node_s, node_v = node_features
        edge_s, edge_v = edge_features
        B, N = padding_mask.shape
        
        #process node features
        self.batch.x = coords[:, :, 1, :].reshape(B * N, -1)   # Just CA
        self.batch.h = node_s.reshape(B * N, -1)
        self.batch.atom_type = atom_type.reshape(B * N, -1)
        self.batch.chi = node_v.reshape(B * N, -1, 3)
        self.batch.mask = (~padding_mask).reshape(B * N)
        
        # Process edge features
        num_edges = edge_index.shape[2]  # number of edges per graph
        edge_s = edge_s.view(B * num_edges, -1)  # Reshape edge scalar features
        edge_v = edge_v.view(B * num_edges, edge_v.shape[-2], edge_v.shape[-1])  # Reshape edge vector features
        
        # Adjust edge_index by adding an offset to each graph's node indices
        edge_index = edge_index.transpose(0,1).reshape(2, B * num_edges)
        node_offset = (torch.arange(B, device=coords.device) * N).repeat_interleave(num_edges)[:,None].expand(-1, 2).transpose(0,1)
        node_offset = torch.where(edge_index != -1, node_offset, 0)       
        edge_index = edge_index + node_offset  # Apply node offset to edge_index
        
        # Assign features to the batch object
        self.batch.e = edge_s
        self.batch.xi = edge_v
        self.batch.edge_index = edge_index
        self.batch.batch = torch.arange(B, device=coords.device).repeat_interleave(N)  # Batch vector
        self.batch.num_graphs = self.batch.batch_size = B

        return self.batch

    def get_atom_type(self, seq, atom14_mask):
        atom_indices = get_rc_tensor(rc.RESTYPE_TO_ATOM37_IDX, seq)
        atom_indices = torch.where(atom_indices == -1, 0, atom_indices) #temporaily set ghost atom idx to 0
        atom_indices_one_hot = F.one_hot(atom_indices, num_classes=rc.atom_type_num).float()
        atom_indices_one_hot *= atom14_mask[..., None].expand_as(atom_indices_one_hot)
        atom_types_summed = torch.sum(atom_indices_one_hot, dim = -2)
        return atom_types_summed
    
    def get_node_features(self, coords, padding_mask, atom14_mask):
        # scalar features
        node_scalar_features = dihedrals(coords)

        # vector features
        X_ca = coords[:, :, 1]
        ca_orientations = orientations(X_ca)
        fa_orientations = self.intra_residue_orientations(coords, atom14_mask)

        #for residues w/out CB, overwrite with pseudo CB
        cb_orientations = sidechains(coords)
        no_cb_mask = atom14_mask[:, :, 4] == 0 #use atom14 mask to find positions with no cb
        no_cb_mask = torch.where(padding_mask, False, no_cb_mask) #exclude padded positions from getting pseudo cb
        fa_orientations[:,:,3][no_cb_mask, :] = cb_orientations[no_cb_mask]

        node_vector_features = torch.cat([ca_orientations, fa_orientations], dim=-2)
        return node_scalar_features, node_vector_features


    def intra_residue_orientations(self, coords, atom14_mask):
        X_ca = coords[:, :, 1]
        vectors = []
        atom_positions = [0,2,3,4,5,6,7,8,9,10,11,12,13]

        for atom_pos in atom_positions:
            atom_pos_mask = atom14_mask[:, :, atom_pos][:,:,None].expand(-1, -1, 3)
            intra_residue_vector = normalize(X_ca - coords[:, :, atom_pos])

            #set unit vector for missing atoms to 0
            if self.zero_ghost_atoms:
                intra_residue_vector = torch.where(atom_pos_mask == 1, intra_residue_vector, 0)
            vectors.append(intra_residue_vector)

        return torch.stack(vectors, dim=2)

    def get_edge_features(self, coords, padding_mask, E_idx, atom14_mask):
        X_ca = coords[:, :, 1]

        # Get distances to the top k neighbors, using E_idx from ProteinMPNN
        E_dist, E_residue_mask = dist(X_ca, E_idx, padding_mask)

        # Flatten the graph to be batch size 1 for torch_geometric package 
        dest = E_idx
        B, L, k = E_idx.shape[:3]
        src = torch.arange(L, device=E_idx.device).view([1, L, 1]).expand(B, L, k)
        # After flattening, [2, B, E]
        edge_index = torch.stack([src, dest], dim=0).flatten(2, 3)
        # After flattening, [B, E]
        E_dist = E_dist.flatten(1, 2)
        E_residue_mask = E_residue_mask.flatten(1, 2)
        # Calculate relative positional embeddings and distance RBF 
        pos_embeddings = positional_embeddings(
            edge_index,
            num_positional_embeddings=self.num_positional_embeddings,
        )
        D_rbf = rbf(E_dist, 0., 20.)
        
        # Calculate relative orientation 
        E_vectors = self.get_edge_vectors(coords, E_idx, edge_index, B, L, k, atom14_mask)

        # Normalize and remove nans 
        edge_s = torch.cat([D_rbf, pos_embeddings], dim=-1)
        edge_v = normalize(E_vectors)
        edge_s, edge_v = map(nan_to_num, (edge_s, edge_v))
        edge_index[:, ~E_residue_mask] = -1

        return (edge_s, edge_v), edge_index.transpose(0, 1) 
    
    def get_edge_vectors(self, coords, E_idx, edge_index, B, L, k, atom14_mask):
        max_atoms = coords.shape[-2] #14
        X_n = coords[:, :, 0]
        X_ca = coords[:, :, 1]
        X_c = coords[:, :, 2]
        X_o = coords[:, :, 3]
        
        vectors = []
        
        for bb_atom in [X_ca, X_n, X_c, X_o]:
            for atom_pos in range(max_atoms):
                atom_mask_pos = atom14_mask[:,:,atom_pos]
                atom_mask_neighbors = torch.gather(atom_mask_pos[...,None].expand(-1,-1,k), 1, E_idx)
                relative_orientation_vector = normalize(self.get_relative_orientation(bb_atom, coords[:,:,atom_pos], edge_index, B, L, k))

                #insert 0 for unit vectors where destination atom does not exist
                #relative_orientation_vector = torch.where(atom_mask_neighbors[...,None].expand(-1,-1,-1,3) == 1, relative_orientation_vector, 0).flatten(1, 2)
                vectors.append(relative_orientation_vector)
        
        return torch.stack(vectors, dim=2)

    def get_relative_orientation(self, X, Y, edge_index, B, L, k):
        X_src = X.unsqueeze(2).expand(-1, -1, k, -1).flatten(1, 2)
        X_dest = torch.gather(
            Y,
            1,
            edge_index[1, :, :].unsqueeze(-1).expand([B, L*k, 3])
        )

        return X_src - X_dest
    
if __name__ == "__main__":
    pass