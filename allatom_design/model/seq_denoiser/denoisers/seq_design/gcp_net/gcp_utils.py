# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from typing import Callable, Optional, Union, Tuple, Any
from copy import copy
from torch_geometric.data import Data
from functools import partial
from torchtyping import TensorType
from typeguard import typechecked

class ScalarVector(tuple):
    """
    From https://github.com/sarpaykent/GBPNet
    """
    def __new__(cls, scalar, vector):
        return tuple.__new__(cls, (scalar, vector))

    def __getnewargs__(self):
        return self.scalar, self.vector

    @property
    def scalar(self):
        return self[0]

    @property
    def vector(self):
        return self[1]

    # Element-wise addition
    def __add__(self, other):
        if isinstance(other, tuple):
            scalar_other = other[0]
            vector_other = other[1]
        else:
            scalar_other = other.scalar
            vector_other = other.vector

        return ScalarVector(self.scalar + scalar_other, self.vector + vector_other)

    # Element-wise multiplication or scalar multiplication
    def __mul__(self, other):
        if isinstance(other, tuple):
            other = ScalarVector(other[0], other[1])

        if isinstance(other, ScalarVector):
            return ScalarVector(self.scalar * other.scalar, self.vector * other.vector)
        else:
            return ScalarVector(self.scalar * other, self.vector * other)

    def concat(self, others, dim=-1):
        dim %= len(self.scalar.shape)
        s_args, v_args = list(zip(*(self, *others)))
        return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)

    def flatten(self):
        flat_vector = torch.reshape(self.vector, self.vector.shape[:-2] + (3 * self.vector.shape[-2],))
        return torch.cat((self.scalar, flat_vector), dim=-1)

    @staticmethod
    def recover(x, vector_dim):
        v = torch.reshape(x[..., -3 * vector_dim:], x.shape[:-1] + (vector_dim, 3))
        s = x[..., : -3 * vector_dim]
        return ScalarVector(s, v)

    def vs(self):
        return self.scalar, self.vector

    def idx(self, idx):
        return ScalarVector(self.scalar[idx], self.vector[idx])

    def repeat(self, n, c=1, y=1):
        return ScalarVector(self.scalar.repeat(n, c), self.vector.repeat(n, y, c))

    def clone(self):
        return ScalarVector(self.scalar.clone(), self.vector.clone())

    def mask(self, node_mask: TensorType["num_nodes"]):
        return ScalarVector(
            self.scalar * node_mask[:, None],
            self.vector * node_mask[:, None, None]
        )

    def __setitem__(self, key, value):
        self.scalar[key] = value.scalar
        self.vector[key] = value.vector

    def __repr__(self):
        return f"ScalarVector({self.scalar}, {self.vector})"


class VectorDropout(nn.Module):
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, drop_rate):
        super(VectorDropout, self).__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        device = x[0].device
        if not self.training:
            return x
        mask = torch.bernoulli((1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class GCPDropout(nn.Module):
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, drop_rate: float):
        super(GCPDropout, self).__init__()
        self.scalar_dropout = nn.Dropout(drop_rate)
        self.vector_dropout = VectorDropout(drop_rate)

    def forward(self, x: Union[torch.Tensor, ScalarVector]):
        if isinstance(x, torch.Tensor) and x.shape[0] == 0:
            return x
        elif isinstance(x, ScalarVector) and (x.scalar.shape[0] == 0 or x.vector.shape[0] == 0):
            return x
        elif isinstance(x, torch.Tensor):
            return self.scalar_dropout(x)
        return ScalarVector(self.scalar_dropout(x[0]), self.vector_dropout(x[1]))


class GCPLayerNorm(nn.Module):
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, dims: ScalarVector, eps: float = 1e-8):
        super(GCPLayerNorm, self).__init__()
        self.scalar_dims, self.vector_dims = dims
        self.scalar_norm = nn.LayerNorm(self.scalar_dims)
        self.eps = eps

    @staticmethod
    def norm_vector(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        vector_norm = torch.clamp(torch.sum(torch.square(v), dim=-1, keepdim=True), min=eps)
        vector_norm = torch.sqrt(torch.mean(vector_norm, dim=-2, keepdim=True))
        return v / vector_norm

    def forward(self, x: Union[torch.Tensor, ScalarVector]):
        if isinstance(x, torch.Tensor) and x.shape[0] == 0:
            return x
        elif isinstance(x, ScalarVector) and (x.scalar.shape[0] == 0 or x.vector.shape[0] == 0):
            return x
        elif not self.vector_dims:
            return (
                self.scalar_norm(x[0])
                if isinstance(x, ScalarVector)
                else self.scalar_norm(x)
            )
        s, v = x
        return ScalarVector(self.scalar_norm(s), self.norm_vector(v, eps=self.eps))

@typechecked
def centralize(
    batch: Batch,
    key: str,
    batch_index: torch.Tensor,
    node_mask: Optional[TensorType["batch_num_nodes"]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:  # note: cannot make assumptions on output shape
    if node_mask is not None:
        # derive centroid of each batch element
        dim_size = batch[key].shape[0]  # Get the max index value along the scatter dimension, and add 1
        src = batch[key][node_mask]
        output_scatter_reduce = torch.zeros((dim_size, src.shape[1]), dtype=src.dtype, device=src.device)  # Initialize output with dynamic shape

        entities_centroid = output_scatter_reduce.scatter_reduce(
            src = src,
            index = batch_index[node_mask].unsqueeze(-1).expand_as(src),
            dim=0,
            reduce="mean",
            include_self=False,
        )  # e.g., [batch_size, 3]

        # center entities using corresponding centroids
        entities_centered = batch[key] - (entities_centroid[batch_index] * node_mask.float().unsqueeze(-1))
        masked_values = (
            torch.ones_like(batch[key]) * torch.inf
        )
        values = batch[key][node_mask]
        masked_values[node_mask] = (values - entities_centroid[batch_index][node_mask])
        entities_centered = masked_values

    else:
        # derive centroid of each batch element, and center entities using corresponding centroids
        output_size = batch_index.max().item() + 1  # Get the max index value along the scatter dimension, and add 1
        output_scatter_reduce = torch.zeros((output_size, batch[key].shape[1]), dtype=batch[key].dtype, device = batch[key].device)  # Initialize output with dynamic shape
        entities_centroid = output_scatter_reduce.scatter_reduce(
        src=batch[key],
        index=batch_index.unsqueeze(-1).expand_as(batch[key]),
        dim=0,
        reduce="mean",
        include_self=False,
        )  # e.g., [batch_size, 3]  # e.g., [batch_size, 3]        entities_centered = batch[key] - entities_centroid[batch_index]
        
        entities_centered = batch[key] - entities_centroid[batch_index]

    return entities_centroid, entities_centered


@typechecked
def decentralize(
    batch: Batch,
    key: str,
    batch_index: torch.Tensor,
    entities_centroid: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:  # note: cannot make assumptions on output shape
    if node_mask is not None:
        masked_values = torch.ones_like(batch[key]) * torch.inf
        masked_values[node_mask] = (batch[key][node_mask] + entities_centroid[batch_index])
        entities_centered = masked_values
    else:
        entities_centered = batch[key] + entities_centroid[batch_index]
    return entities_centered

@typechecked
def localize(
    x: TensorType["batch_num_nodes", 3],
    edge_index: TensorType[2, "batch_num_edges"],
    norm_x_diff: bool = True,
    node_mask: Optional[torch.Tensor] = None
) -> TensorType["batch_num_edges", 3, 3]:
    row, col = edge_index[0], edge_index[1]

    if node_mask is not None:
        edge_mask = node_mask[row] & node_mask[col]

        x_diff = torch.ones((edge_index.shape[1], 3), device=edge_index.device) * torch.inf
        x_diff[edge_mask] = x[row][edge_mask] - x[col][edge_mask]

        x_cross = torch.ones((edge_index.shape[1], 3), device=edge_index.device) * torch.inf
        x_cross[edge_mask] = torch.cross(x[row][edge_mask], x[col][edge_mask])
    else:
        x_diff = x[row] - x[col]
        x_cross = torch.cross(x[row], x[col])

    if norm_x_diff:
        # derive and apply normalization factor for `x_diff`
        if node_mask is not None:
            norm = torch.ones((edge_index.shape[1], 1), device=x_diff.device)
            norm[edge_mask] = (
                torch.sqrt(torch.sum((x_diff[edge_mask] ** 2), dim=1).unsqueeze(1))
            ) + 1
        else:
            norm = torch.sqrt(torch.sum((x_diff) ** 2, dim=1).unsqueeze(1)) + 1
        x_diff = x_diff / norm

        # derive and apply normalization factor for `x_cross`
        if node_mask is not None:
            cross_norm = torch.ones((edge_index.shape[1], 1), device=x_cross.device)
            cross_norm[edge_mask] = (
                torch.sqrt(torch.sum((x_cross[edge_mask]) ** 2, dim=1).unsqueeze(1))
            ) + 1
        else:
            cross_norm = (torch.sqrt(torch.sum((x_cross) ** 2, dim=1).unsqueeze(1))) + 1
        x_cross = x_cross / cross_norm

    if node_mask is not None:
        x_vertical = torch.ones((edge_index.shape[1], 3), device=edge_index.device) * torch.inf
        x_vertical[edge_mask] = torch.cross(x_diff[edge_mask], x_cross[edge_mask])
    else:
        x_vertical = torch.cross(x_diff, x_cross)

    f_ij = torch.cat((x_diff.unsqueeze(1), x_cross.unsqueeze(1), x_vertical.unsqueeze(1)), dim=1)
    return f_ij


@typechecked
def scalarize(
    vector_rep: TensorType["batch_num_entities", 3, 3],
    edge_index: TensorType[2, "batch_num_edges"],
    frames: TensorType["batch_num_edges", 3, 3],
    node_inputs: bool,
    enable_e3_equivariance: bool,
    dim_size: int,
    node_mask: Optional[TensorType["batch_num_nodes"]] = None
) -> TensorType["effective_batch_num_entities", 9]:
    row, col = edge_index[0], edge_index[1]

    # gather source node features for each `entity` (i.e., node or edge)
    # note: edge inputs are already ordered according to source nodes
    vector_rep_i = vector_rep[row] if node_inputs else vector_rep

    # project equivariant values onto corresponding local frames
    if vector_rep_i.ndim == 2:
        vector_rep_i = vector_rep_i.unsqueeze(-1)
    elif vector_rep_i.ndim == 3:
        vector_rep_i = vector_rep_i.transpose(-1, -2)

    if node_mask is not None:
        edge_mask = node_mask[row] & node_mask[col]
        local_scalar_rep_i = torch.zeros((edge_index.shape[1], 3, 3), device=edge_index.device, dtype=vector_rep_i.dtype)
        local_scalar_rep_i[edge_mask] = torch.matmul(
            frames[edge_mask], vector_rep_i[edge_mask].to(frames.dtype)
        )
        local_scalar_rep_i = local_scalar_rep_i.transpose(-1, -2)
    else:
        local_scalar_rep_i = torch.matmul(frames, vector_rep_i).transpose(-1, -2)

    # potentially enable E(3)-equivariance and, thereby, chirality-invariance
    if enable_e3_equivariance:
        # avoid corrupting gradients with an in-place operation
        local_scalar_rep_i_copy = local_scalar_rep_i.clone()
        local_scalar_rep_i_copy[:, :, 1] = torch.abs(local_scalar_rep_i[:, :, 1])
        local_scalar_rep_i = local_scalar_rep_i_copy

    # reshape frame-derived geometric scalars
    local_scalar_rep_i = local_scalar_rep_i.reshape(vector_rep_i.shape[0], 9)

    if node_inputs:
        # for node inputs, summarize all edge-wise geometric scalars using an average
        output_scatter_reduce = torch.zeros((dim_size, local_scalar_rep_i.shape[1]), dtype=local_scalar_rep_i.dtype, device = local_scalar_rep_i.device)

        #the -1 index is ignored in torch_scatter.scatter, but for scatter_reduce we have to explicitly remove it
        valid_mask = row != -1
        row_filtered = row[valid_mask]
        local_scalar_rep_i_filtered = local_scalar_rep_i[valid_mask]        

        local_scalar_rep_i =  output_scatter_reduce.scatter_reduce(
            src=local_scalar_rep_i_filtered,
            index=row_filtered.unsqueeze(-1).expand_as(local_scalar_rep_i_filtered),
            dim=0,
            reduce="mean",
            include_self = False,
        )

    return local_scalar_rep_i


@typechecked
def vectorize(
    gate: TensorType["batch_num_entities", 9],
    edge_index: TensorType[2, "batch_num_edges"],
    frames: TensorType["batch_num_edges", 3, 3],
    node_inputs: bool,
    dim_size: int,
    node_mask: Optional[TensorType["batch_num_nodes"]] = None
) -> TensorType["effective_batch_num_entities", 3, 3]:
    row, col = edge_index

    frames = frames.reshape(frames.shape[0], 1, 9)
    x_diff, x_cross, x_vertical = frames[:, :, :3].squeeze(
    ), frames[:, :, 3:6].squeeze(), frames[:, :, 6:].squeeze()

    # gather source node features for each `entity` (i.e., node or edge)
    gate = gate[row] if node_inputs else gate  # note: edge inputs are already ordered according to source nodes

    # derive edge mask if provided node mask
    if node_mask is not None:
        edge_mask = node_mask[row] & node_mask[col]

    # use invariant scalar features to derive new vector features using each neighboring node
    gate_vector = torch.zeros_like(gate)
    for i in range(0, gate.shape[-1], 3):
        if node_mask is not None:
            gate_vector[edge_mask, i:i + 3] = (
                gate[edge_mask, i:i + 1] * x_diff[edge_mask]
                + gate[edge_mask, i + 1:i + 2] * x_cross[edge_mask]
                + gate[edge_mask, i + 2:i + 3] * x_vertical[edge_mask]
            )
        else:
            gate_vector[:, i:i + 3] = (
                gate[:, i:i + 1] * x_diff
                + gate[:, i + 1:i + 2] * x_cross
                + gate[:, i + 2:i + 3] * x_vertical
            )
    gate_vector = gate_vector.reshape(gate_vector.shape[0], 3, 3)

    # for node inputs, summarize all edge-wise geometric vectors using an average
    if node_inputs:
        output_scatter_reduce = torch.zeros(dim_size, dtype=gate_vector.dtype)
        
        return output_scatter_reduce.scatter_reduce(
            src = gate_vector,
            # summarize according to source node indices due to the directional nature of GCP2's equivariant frames
            index = row,
            dim=0,
            dim_size=dim_size,
            reduce="mean",
            include_self = False
        )

    return gate_vector


@typechecked
def safe_norm(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
    keepdim: bool = False,
    sqrt: bool = True
):
    norm = torch.sum(x ** 2, dim=dim, keepdim=keepdim)
    if sqrt:
        norm = torch.sqrt(norm + eps)
    return norm + eps


@typechecked
def is_identity(nonlinearity: Optional[Union[Callable, nn.Module]] = None):
    return nonlinearity is None or isinstance(nonlinearity, nn.Identity)


@typechecked
def norm_no_nan(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    eps: float = 1e-8,
    sqrt: bool = True
):
    """
    From https://github.com/drorlab/gvp-pytorch

    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    """
    out = torch.clamp(torch.sum(torch.square(x), dim=dim, keepdim=keepdim), min=eps)
    return torch.sqrt(out) if sqrt else out

@typechecked
def get_nonlinearity(nonlinearity: Optional[str] = None, slope: float = 1e-2, return_functional: bool = False) -> Any:
    nonlinearity = nonlinearity if nonlinearity is None else nonlinearity.lower().strip()
    if nonlinearity == "relu":
        return F.relu if return_functional else nn.ReLU()
    elif nonlinearity == "leakyrelu":
        return partial(F.leaky_relu, negative_slope=slope) if return_functional else nn.LeakyReLU(negative_slope=slope)
    elif nonlinearity == "selu":
        return partial(F.selu) if return_functional else nn.SELU()
    elif nonlinearity == "silu":
        return partial(F.silu) if return_functional else nn.SiLU()
    elif nonlinearity == "sigmoid":
        return torch.sigmoid if return_functional else nn.Sigmoid()
    elif nonlinearity is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f"The nonlinearity {nonlinearity} is currently not implemented.")