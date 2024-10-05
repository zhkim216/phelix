import torch as pt
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# >> MODEL

class GraphTransformer(pt.nn.Module):
    def __init__(self, config):
        super(GraphTransformer, self).__init__()
        # features encoding models for structures and library
        self.em = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        
        # atomic level state update model
        sum_layers = sum([ 
            [{'Ns':config.hidden_dim,
            'Nh':config.num_heads, 
            'Nk':config.dim_key, 
            'pos_enc':config.pos_enc,
            'attn_bias':config.attn_bias,
            'dim_pos_enc':config.dim_pos_enc,
            'nn': nn }] * config.layers_per_nn for nn in config.nns 
            ], [])

        self.sum = nn.Sequential(*[StateUpdateLayer(layer_params) for layer_params in sum_layers])

        # decoding mlp
        self.dm = nn.Sequential(
            nn.Linear(2*config.hidden_dim,config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.out_dim),
        )

    def forward(self, X, ids_topk, q0, p_A, attn_bias):
        # encode features
        q = self.em.forward(q0)

        # initial state vectors
        p0 = pt.zeros((q.shape[0]+1, X.shape[1], q.shape[1]), device=X.device)

        # unpack state features with sink
        q, ids_topk, D_nn, R_nn, p_A, attn_bias = unpack_state_features(X, ids_topk, q, p_A, attn_bias)

        # atomic tsa layers
        qa, pa, _, _, _, _, _ = self.sum.forward((q, p0, ids_topk, D_nn, R_nn, p_A, attn_bias))

        z = pt.cat([qa[1:], pt.norm(pa[1:], dim=1)], dim=1)

        z = self.dm.forward(z)

        return z


# >> UTILS
def unpack_state_features(X, ids_topk, q, p_A, attn_bias):
    # compute displacement vectors
    R_nn = X[ids_topk-1] - X.unsqueeze(1)
    # compute distance matrix
    D_nn = pt.norm(R_nn, dim=2)
    # mask distances
    D_nn = D_nn + pt.max(D_nn)*(D_nn < 1e-2).float()
    # normalize displacement vectors
    R_nn = R_nn / D_nn.unsqueeze(2)

    # prepare sink
    q = pt.cat([pt.zeros((1, q.shape[1]), device=q.device), q], dim=0)
    ids_topk = pt.cat([pt.zeros((1, ids_topk.shape[1]), dtype=pt.long, device=ids_topk.device), ids_topk], dim=0)
    if p_A is not None:
        p_A = pt.cat([pt.zeros((1, p_A.shape[1], p_A.shape[2]), device=p_A.device), p_A], dim=0)
    if attn_bias is not None:
        attn_bias = pt.cat([pt.zeros((1, attn_bias.shape[1], attn_bias.shape[2]), device=attn_bias.device), attn_bias], dim=0)
    D_nn = pt.cat([pt.zeros((1, D_nn.shape[1]), device=D_nn.device), D_nn], dim=0)
    R_nn = pt.cat([pt.zeros((1, R_nn.shape[1], R_nn.shape[2]), device=R_nn.device), R_nn], dim=0)

    return q, ids_topk, D_nn, R_nn, p_A, attn_bias


# >>> OPERATIONS
class StateUpdate(pt.nn.Module):
    def __init__(self, Ns, Nh, Nk, pos_enc, attn_bias, dim_pos_enc):
        super(StateUpdate, self).__init__()
        # operation parameters
        self.Ns = Ns
        self.Nh = Nh
        self.Nk = Nk
        self.pos_enc = pos_enc
        self.attn_bias = attn_bias

        if not self.pos_enc:
            dim_pos_enc = 0

        # node query model
        self.nqm = pt.nn.Sequential(
            pt.nn.Linear(2*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, 2*Nk*Nh),
        )

        # edges scalar keys model
        self.eqkm = pt.nn.Sequential(
            pt.nn.Linear(6*Ns+1+dim_pos_enc, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Nk),
        )

        # edges vector keys model
        self.epkm = pt.nn.Sequential(
            pt.nn.Linear(6*Ns+1+dim_pos_enc, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, 3*Nk),
        )

        # edges value model
        self.evm = pt.nn.Sequential(
            pt.nn.Linear(6*Ns+1+dim_pos_enc, 2*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(2*Ns, 2*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(2*Ns, 2*Ns),
        )

        # scalar projection model
        self.qpm = pt.nn.Sequential(
            pt.nn.Linear(Nh*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
        )

        # vector projection model
        self.ppm = pt.nn.Sequential(
            pt.nn.Linear(Nh*Ns, Ns, bias=False),
        )

        # scaling factor for attention
        self.sdk = pt.nn.Parameter(pt.sqrt(pt.tensor(Nk).float()), requires_grad=False)

    def forward(self, q, p, q_nn, p_nn, D_topk, R_topk, p_A, attn_bias, m_nn):
        # q: [N, S]
        # p: [N, 3, S]
        # q_nn: [N, n, S]
        # p_nn: [N, n, 3, S]
        # d_nn: [N, n]
        # r_nn: [N, n, 3]
        # p_A: [N, n]
        # N: number of nodes
        # n: number of nearest neighbors
        # S: state dimensions
        # H: number of attention heads

        # get dimensions
        d_nn, r_nn = D_topk[:,m_nn], R_topk[:,m_nn]
        N, n, S = q_nn.shape

        # node inputs packing
        X_n = pt.cat([
            q,
            pt.norm(p, dim=1),
        ], dim=1)  # [N, 2*S]

        # edge inputs packing
        X_e = pt.cat([
            d_nn.unsqueeze(2),                                  # distance
            X_n.unsqueeze(1).repeat(1,n,1),                     # centered state
            q_nn,                                               # neighbors states
            pt.norm(p_nn, dim=2),                               # neighbors vector states norms
            pt.sum(p.unsqueeze(1) * r_nn.unsqueeze(3), dim=2),  # centered vector state projections
            pt.sum(p_nn * r_nn.unsqueeze(3), dim=2),            # neighbors vector states projections
        ], dim=2)  # [N, n, 6*S+1]

        if self.pos_enc and p_A is not None:
            p_A = p_A[:, m_nn]
            X_e = pt.cat([X_e, p_A], dim=2)

        # node queries
        Q = self.nqm.forward(X_n).view(N, 2, self.Nh, self.Nk)  # [N, 2*S] -> [N, 2, Nh, Nk]

        # scalar edges keys while keeping interaction order inveriance
        Kq = self.eqkm.forward(X_e).view(N, n, self.Nk).transpose(1,2)  # [N, n, 6*S+1] -> [N, Nk, n]

        # vector edges keys while keeping bond order inveriance
        Kp = pt.cat(pt.split(self.epkm.forward(X_e), self.Nk, dim=2), dim=1).transpose(1,2)

        # edges values while keeping interaction order inveriance
        V = self.evm.forward(X_e).view(N, n, 2, S).transpose(1,2)  # [N, n, 6*S+1] -> [N, 2, n, S]

        # vectorial inputs packing
        Vp = pt.cat([
            V[:,1].unsqueeze(2) * r_nn.unsqueeze(3),
            p.unsqueeze(1).repeat(1,n,1,1),
            p_nn,
            #pt.cross(p.unsqueeze(1).repeat(1,n,1,1), r_nn.unsqueeze(3).repeat(1,1,1,S), dim=2),
        ], dim=1).transpose(1,2)  # [N, 3, 3*n, S]

        # queries and keys collapse
        if self.attn_bias and attn_bias is not None:
            attn_bias = attn_bias[:, :, m_nn]
            Mq = pt.nn.functional.softmax(pt.matmul(Q[:,0], Kq) + attn_bias / self.sdk, dim=2)  # [N, Nh, n]
            Mp = pt.nn.functional.softmax(pt.matmul(Q[:,1], Kp) + attn_bias.repeat(1, 1, 3) / self.sdk, dim=2)  # [N, Nh, 3*n]
        else:
            Mq = pt.nn.functional.softmax(pt.matmul(Q[:,0], Kq) / self.sdk, dim=2)  # [N, Nh, n]
            Mp = pt.nn.functional.softmax(pt.matmul(Q[:,1], Kp)  / self.sdk, dim=2)  # [N, Nh, 3*n] 

        # scalar state attention mask and values collapse
        Zq = pt.matmul(Mq, V[:,0]).view(N, self.Nh*self.Ns)  # [N, Nh*S]
        Zp = pt.matmul(Mp.unsqueeze(1), Vp).view(N, 3, self.Nh*self.Ns)  # [N, 3, Nh*S]

        # decode outputs
        qh = self.qpm.forward(Zq)
        ph = self.ppm.forward(Zp)

        # update state with residual
        qz = q + qh
        pz = p + ph

        return qz, pz


# >>> LAYERS
class StateUpdateLayer(pt.nn.Module):
    def __init__(self, layer_params):
        super(StateUpdateLayer, self).__init__()
        # define operation
        self.su = StateUpdate(*[layer_params[k] for k in ['Ns', 'Nh', 'Nk','pos_enc','attn_bias','dim_pos_enc']])
        # store number of nearest neighbors
        self.m_nn = pt.nn.Parameter(pt.arange(layer_params['nn'], dtype=pt.int64), requires_grad=False)

    def forward(self, Z):
        # unpack input
        q, p, ids_topk, D_topk, R_topk, p_A, attn_bias = Z

        # update q, p
        ids_nn = ids_topk[:,self.m_nn]

        # with checkpoint
        q = q.requires_grad_()
        p = p.requires_grad_()

        q, p = checkpoint(self.su.forward, q, p, q[ids_nn], p[ids_nn], D_topk, R_topk, p_A, attn_bias, self.m_nn)

        # sink
        q[0] = q[0] * 0.0
        p[0] = p[0] * 0.0

        return q, p, ids_topk, D_topk, R_topk, p_A, attn_bias