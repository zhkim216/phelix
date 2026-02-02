import math
from functools import partial
from typing import Optional, Union
from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn as nn
import torch._dynamo as dynamo
from torch.nn import functional as F
from torchtyping import TensorType

import allatom_design.data.const as const
import allatom_design.model.seq_denoiser.denoisers.seq_design.potts as potts
from allatom_design.data.data import batched_gather
from allatom_design.model.seq_denoiser.denoisers.seq_design.mpnn_utils import (
    cat_neighbors_nodes, gather_edges, gather_nodes)
from allatom_design.data.const import PERIODIC_TABLE_FEATURES
from atomworks.constants import (
    DNA_BACKBONE_ATOM_NAMES,
    ELEMENT_NAME_TO_ATOMIC_NUMBER,
    METAL_ELEMENTS,
    NUCLEIC_ACID_BACKBONE_ATOM_NAMES,
    PROTEIN_BACKBONE_ATOM_NAMES,
    RNA_BACKBONE_ATOM_NAMES,
    STANDARD_AA,
    STANDARD_AA_TIP_ATOM_NAMES,
    STANDARD_DNA,
    STANDARD_PURINE_RESIDUES,
    STANDARD_PYRIMIDINE_RESIDUES,
    STANDARD_RNA,
    UNKNOWN_AA,
    UNKNOWN_DNA,
    UNKNOWN_RNA,
)


class AtomMPNN(nn.Module):
    """Modified ProteinMPNN network to predict sequence from full atom structure."""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.ligand_conditioning = cfg.ligand_conditioning
        self.node_features = cfg.n_channel
        self.edge_features = cfg.n_channel
        self.hidden_dim = cfg.n_channel
        self.num_encoder_layers = cfg.n_layers
        self.num_decoder_layers = cfg.n_layers
        self.k_neighbors = cfg.k_neighbors
        self.n_tokens = const.AF3_ENCODING.n_tokens        
                        
        self.token_features = TokenFeatures(cfg.token_features)
        self.W_e = nn.Linear(self.edge_features, self.hidden_dim, bias=False)
        self.W_s = nn.Linear(self.n_tokens, self.hidden_dim, bias=False)
        self.decoder_in = self.hidden_dim * 3  # concat of h_E, h_S, h_V

        self.dropout = nn.Dropout(cfg.dropout_p)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(self.hidden_dim, self.hidden_dim*2, dropout=cfg.dropout_p)
            for i in range(self.num_encoder_layers)
        ])
        
        if self.ligand_conditioning:
            cfg_lmpnn_module = cfg.get("lmpnn_module", None)
            if cfg_lmpnn_module is None:
                cfg_lmpnn_module = {"num_context_feature_processor_layers": 2, "num_context_feature_aggregator_layers": 2, "edge_update": False}
        
            self.num_context_feature_processor_layers = cfg_lmpnn_module.get("num_context_feature_processor_layers", 2)
            self.num_context_feature_aggregator_layers = cfg_lmpnn_module.get("num_context_feature_aggregator_layers", 2)
            self.edge_update = cfg_lmpnn_module.get("edge_update", False)

            # Encapsulate context feature processing into a separate module
            self.context_module = ContextModule(
                hidden_dim=self.hidden_dim,
                dropout_p=cfg.dropout_p,
                num_processor_layers=self.num_context_feature_processor_layers,
                num_aggregator_layers=self.num_context_feature_aggregator_layers,
                edge_update=self.edge_update,
            )
            
                        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(self.hidden_dim, self.decoder_in, dropout=cfg.dropout_p)
            for _ in range(self.num_decoder_layers)
        ])

        # Potts decoder
        self.use_potts = cfg.potts.use_potts
        if self.use_potts:
            self.k_neighbors_potts = cfg.potts.get("k_neighbors_potts", None)
            self.max_dist_potts = cfg.potts.get("max_dist_potts", None)
            self.parameterization = cfg.potts.parameterization
            self.num_factors = cfg.potts.num_factors

            potts_init = partial(potts.GraphPotts,
                dim_nodes=self.node_features,
                dim_edges=self.decoder_in,
                num_states=self.n_tokens,
                parameterization=self.parameterization,
                num_factors=self.num_factors,
                symmetric_J=cfg.potts.symmetric_J,
                dropout=cfg.dropout_p,
            )
            self.decoder_S_potts = potts_init()

        # Output layers
        self.W_out = nn.Linear(self.hidden_dim, self.n_tokens, bias=True)

        # Initialize weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, batch: dict[str, TensorType["b ..."]], is_sampling: bool):        
        # Get token-level features
        B, N, C = batch["restype"].shape
        h_V = torch.zeros((B, N, self.node_features), device=batch["restype"].device)

        # Concatenate residue-level features to h_V
        ## first, mask out residues using gap token        
        masked = F.one_hot(torch.full((B, N), const.AF3_ENCODING.token_to_idx["<G>"],
                                      device=batch["restype"].device), num_classes=C).float()
        
        #! (JH) During sampling, seq_cond_mask is also 1 for padded tokens
        #! (JH) So padded parts are also considered as gaps here, but I guess it's okay.        
        restype = torch.where(batch["seq_cond_mask"].unsqueeze(-1).bool(), batch["restype"], masked)
        h_S = self.W_s(restype) #! (JH) different from the original lmpnn (zero-initialized)
        

        # Build graph and get edge features
        h_E, E_idx, V, Y_nodes, Y_edges, Y_m, D_neighbors = self.token_features(batch)
        #! (JH) h_E and E_idx are also considering ligand atoms here.
        #! (JH) but h_E and E_idx are masked out for padded tokens (token_exists_mask is 0 for padded tokens)        
                                        
        # Pass through encoder layers        
        # Residue-level encoding, for standard AAs in protein chains only
        h_V = h_V + h_S                
        h_E = self.W_e(h_E)        
                
        protein_residue_node_mask = batch["protein_residue_node_mask"]
        protein_residue_node_mask_2d = gather_nodes(protein_residue_node_mask.unsqueeze(-1), E_idx).squeeze(-1)
        protein_residue_node_mask_2d = protein_residue_node_mask.unsqueeze(-1) * protein_residue_node_mask_2d
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, protein_residue_node_mask, protein_residue_node_mask_2d)        
                    
        # Process ligand context features
        if self.ligand_conditioning:
            h_V = self.context_module(
                h_V=h_V,
                h_E=h_E,
                V=V,                 
                Y_nodes=Y_nodes,
                Y_edges=Y_edges,
                Y_m=Y_m,
                E_idx=E_idx,
                protein_residue_node_mask=protein_residue_node_mask,
            )
            
        # Pass through decoder layers
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
        
        for layer in self.decoder_layers:
            h_V, h_ESV = layer(h_V = h_V, h_E = h_ESV, mask_V = protein_residue_node_mask, E_idx = E_idx, mask_attend = protein_residue_node_mask_2d) 

        # Potts model
        if self.use_potts:
            if self.max_dist_potts is not None:
                protein_residue_node_mask_2d = protein_residue_node_mask_2d * (D_neighbors <= self.max_dist_potts)  # mask out edges that are too far away

            if self.k_neighbors_potts is not None:
                # truncate to k_neighbors_potts
                h_ESV = h_ESV[:, :, :self.k_neighbors_potts]
                E_idx = E_idx[:, :, :self.k_neighbors_potts]
                protein_residue_node_mask_2d = protein_residue_node_mask_2d[:, :, :self.k_neighbors_potts]

            h, J = self.decoder_S_potts(h_V, h_ESV, E_idx, protein_residue_node_mask, protein_residue_node_mask_2d)
            potts_decoder_aux = {
                "h": h,
                "J": J,
                "edge_idx": E_idx,
                "mask_i": protein_residue_node_mask,
                "mask_ij": protein_residue_node_mask_2d,
            }

        logits = self.W_out(h_V)                

        # Output features
        mpnn_feature_dict = {"h_V": h_V, "h_ESV": h_ESV, "E_idx": E_idx}
        if self.use_potts:
            mpnn_feature_dict["potts_decoder_aux"] = potts_decoder_aux

        return logits, mpnn_feature_dict


class TokenFeatures(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        Extract token-level edge features and build KNN graph.
        And also extract ligand-related features if ligand_conditioning is True.
        """
        super().__init__()
        self.cfg = cfg

        # Parameters        
        self.k_neighbors = cfg.k_neighbors        
        self.num_positional_embeddings = cfg.num_positional_embeddings
        self.node_n_channel = cfg.node_n_channel
        self.edge_n_channel = cfg.edge_n_channel    
        
        # Positional embeddings
        self.positional_embeddings = PositionalEncodings(self.num_positional_embeddings)
        
        # RBF-related parameters
        self.num_rbf = cfg.num_rbf
        self.min_rbf_mean = cfg.min_rbf_mean
        self.max_rbf_mean = cfg.max_rbf_mean
        
        # Protein graph-related parameters
        self.protein_graph_rbf_type = cfg.protein_graph_rbf_type
        if self.protein_graph_rbf_type == "ca":
            num_pairwise_dists = 1
        elif self.protein_graph_rbf_type == "ncaco":
            num_pairwise_dists = 4*4
        elif self.protein_graph_rbf_type == "ncacocb":
            num_pairwise_dists = 5*5
        protein_graph_edge_in = self.num_positional_embeddings + self.num_rbf * num_pairwise_dists                
        self.protein_edge_embedding = nn.Linear(protein_graph_edge_in, self.edge_n_channel, bias=False)                                
        self.norm_protein_edges = nn.LayerNorm(self.edge_n_channel)
                
        # Context-related parameters
        self.use_multichain_encoding = cfg.get("use_multichain_encoding", True)
        self.ligand_conditioning = cfg.ligand_conditioning
        self.use_sidechain_context = cfg.get("use_sidechain_context", True)
        self.use_ligand_context = cfg.get("use_ligand_context", True)
        self.sidechain_context_token_num = cfg.get("sidechain_context_token_num", 16)
        self.ligand_atom_context_num = cfg.get("ligand_atom_context_num", 16)                                
        
        # Ligand conditioning-related layers
        if self.ligand_conditioning:
            self.protein_ligand_interaction_rbf_type = cfg.get("protein_ligand_interaction_rbf_type", "ncacocb")
            if self.protein_ligand_interaction_rbf_type == "cb":
                num_prot_anchor_atoms = 1
            elif self.protein_ligand_interaction_rbf_type == "ncacocb":
                num_prot_anchor_atoms = 5
            
            # Linear layer for atom type information embedding
            self.type_linear = torch.nn.Linear(147, 64) 

            # Parameters for Ligand-protein interaction layers
            self.add_angle_features = cfg.get("add_angle_features", True)            
            num_angle_features = 4 if self.add_angle_features else 0
            
            self.node_project_down = torch.nn.Linear(
            self.num_rbf * num_prot_anchor_atoms + 64 + num_angle_features, self.node_n_channel, bias=True
        )
            self.norm_nodes = torch.nn.LayerNorm(self.node_n_channel)
            
            # Parameters for Ligand subgraph
            # ligand subgraph nodes
            self.y_nodes = torch.nn.Linear(147, self.node_n_channel, bias=False)
            self.norm_y_nodes = torch.nn.LayerNorm(self.node_n_channel)
            
            # ligand subgraph edges
            self.y_edges = torch.nn.Linear(self.num_rbf, self.edge_n_channel, bias=False)
            self.norm_y_edges = torch.nn.LayerNorm(self.edge_n_channel)
            
                                
    def forward(self, batch: dict[str, TensorType["b ..."]]):
        """
        Extract token-level edge features and build KNN graph.
        """
        # calculate n, ca, c, o and pseudo CB coordinates                            
        X = self._get_protein_token_center_coords(batch) # CA coordinates for protein tokens
        D_neighbors, E_idx = self._dist(X = X, mask = batch["protein_residue_node_mask"]) 
                        
        # Get RBF features
        if self.protein_graph_rbf_type == "ca":
            RBF_backbone = self._rbf(D_neighbors)
        else:            
            RBF_backbone = self.get_backbone_pseudocb_rbf(batch = batch, D_neighbors = D_neighbors, E_idx = E_idx, rbf_type = self.protein_graph_rbf_type)
            
        # Positional encodings
        residue_index = batch["residue_index"]
        offset = residue_index[:,:,None] - residue_index[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0]  # [B, L, K] # Gathering only edges between protein tokens        

        # Chain information
        chain_labels = torch.zeros_like(batch["asym_id"])        
        chain_labels = batch["asym_id"]
        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long()  # find self vs non-self interaction
        
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]        
        E_positional = self.positional_embeddings(offset.long(), E_chains) # Get positional encodings for edges only between protein tokens
    
        # Concatenate edge features and embed
        E = torch.cat((E_positional, RBF_backbone), -1)
                
        E = self.protein_edge_embedding(E)
        E = self.norm_protein_edges(E)
        
        if self.ligand_conditioning:                        
            ######### Prepare necessary tensors #########                        
            # Protein-related tensors                        
            # Noised coordinates
            noised_coords = batch["noised_coords"]            
            noised_ca_coords = batch["noised_ca_coords"]
            noised_n_coords = batch["noised_n_coords"]
            noised_c_coords = batch["noised_c_coords"]
            noised_o_coords = batch["noised_o_coords"]
            noised_pseudo_cb_coords = batch["noised_pseudo_cb_coords"]            
            noised_backbone_pseudo_cb_coords = torch.cat((noised_n_coords[:, :, None, :],                                                          
                                                          noised_ca_coords[:, :, None, :],
                                                          noised_c_coords[:, :, None, :],
                                                          noised_o_coords[:, :, None, :],
                                                          noised_pseudo_cb_coords[:, :, None, :]), dim=2)
            
            # Token-level coordinates
            tokenwise_noised_coords = get_tokenwise_coords(noised_coords, batch["tokenwise_atom_idxs"], batch["tokenwise_atom_idxs_mask"])
            
            # Protein residue mask (standard aa in protein chains)
            protein_residue_node_mask = batch["protein_residue_node_mask"]
            atom_is_standard_aa_protein = batch["protein_residue_node_mask"].gather(dim=-1, index=batch["atom_to_token_map"]) * batch["atom_pad_mask"] * batch["atom_resolved_mask"]
                                    
            ##### Ligand context-related tensors (including small molecule, nucleic acid, metals)
            atom_is_not_standard_aa_protein = (1 - atom_is_standard_aa_protein) * batch["atom_pad_mask"] * batch["atom_resolved_mask"]
            ligand_mask = atom_is_not_standard_aa_protein * batch["atom_cond_mask"]            
            noised_ligand_coords = noised_coords * ligand_mask.unsqueeze(-1)
            ligand_atomic_number = batch["atomic_number"] * ligand_mask
                                                
            ## Sidechain context processing
            if self.use_sidechain_context:
                E_idx_sub = E_idx[:, :, :self.sidechain_context_token_num] #! hardcoded for 16 neighbor tokens            
                                
                # Sidechain mask
                scn_mask = batch["prot_scn_wo_cb_atom_mask"] * atom_is_standard_aa_protein * batch["atom_cond_mask"] #! important, CB is not included in sidechain atoms
                tokenwise_scn_mask = batched_gather(scn_mask, batch["tokenwise_atom_idxs"], dim=1, no_batch_dims=1) # xyz_37_m
                tokenwise_scn_mask = tokenwise_scn_mask * batch["tokenwise_atom_idxs_mask"]
                R_m = gather_nodes(tokenwise_scn_mask, E_idx_sub)
                R_m = R_m.view(R_m.shape[0], R_m.shape[1], -1)
                            
                # Sidechain coordinates
                X_scn = tokenwise_noised_coords * tokenwise_scn_mask.unsqueeze(-1)
                R = gather_nodes(X_scn.view(X_scn.shape[0], X_scn.shape[1], -1), E_idx_sub).view(X_scn.shape[0], X_scn.shape[1], E_idx_sub.shape[2], -1, 3)            
                #! all masked atoms' coordinates are set to 0.
                # Todo: set to center atom coordinates maybe?
                R = R.view(X_scn.shape[0], X_scn.shape[1], -1, 3)
                                        
                # Sidechain atomic number
                scn_atomic_number = batch["atomic_number"] * scn_mask                
                tokenwise_scn_atomic_number = batched_gather(scn_atomic_number, batch["tokenwise_atom_idxs"], dim=1, no_batch_dims=1) * batch["tokenwise_atom_idxs_mask"]
                R_t = gather_nodes(tokenwise_scn_atomic_number, E_idx_sub)
                R_t = R_t.view(X_scn.shape[0], X_scn.shape[1], -1) 
            
            else:
                R_m = torch.zeros(batch["pseudo_cb_coords"].shape[0], batch["pseudo_cb_coords"].shape[1], self.sidechain_context_token_num * const.MAX_NUM_ATOMS, device=batch["coords"].device)
                R_t = torch.zeros(batch["pseudo_cb_coords"].shape[0], batch["pseudo_cb_coords"].shape[1], self.sidechain_context_token_num * const.MAX_NUM_ATOMS, device=batch["coords"].device)
                R = torch.zeros(batch["pseudo_cb_coords"].shape[0], batch["pseudo_cb_coords"].shape[1], self.sidechain_context_token_num * const.MAX_NUM_ATOMS, 3, device=batch["coords"].device)
            
            ## Ligand information aggregation 
            # Prepare ligand information                                    
            if self.use_ligand_context:
                Y, Y_t, Y_m, D_XY = self._get_nearest_ligand_atoms(CB = noised_pseudo_cb_coords,
                                                                mask = protein_residue_node_mask,
                                                                Y = noised_ligand_coords,
                                                                Y_t = ligand_atomic_number,
                                                                Y_m = ligand_mask,
                                                                number_of_ligand_atoms = self.ligand_atom_context_num,
                                                                device = batch["coords"].device, 
                                                                ) 
            else:
                Y = torch.zeros(batch["pseudo_cb_coords"].shape[0], batch["pseudo_cb_coords"].shape[1], self.ligand_atom_context_num, 3, device=batch["coords"].device)
                Y_t = torch.zeros(batch["pseudo_cb_coords"].shape[0], batch["pseudo_cb_coords"].shape[1], self.ligand_atom_context_num, device=batch["coords"].device)
                Y_m = torch.zeros(batch["pseudo_cb_coords"].shape[0], batch["pseudo_cb_coords"].shape[1], self.ligand_atom_context_num, device=batch["coords"].device)
                                                                                                                               
            ## Concetenate sidechain and ligand information
            # Masks
            Y_m = torch.cat((R_m, Y_m), dim=2).to(dtype=torch.long) # concat sidechain and ligand masks
            
            # Coordinates                                                            
            Y = torch.cat((R, Y), dim=2) # concat sidechain and ligand coordinates
                              
            # Atomic numbers
            Y_t = torch.cat((R_t, Y_t), dim=2) # concat sidechain and ligand atomic numbers
            
            # Pairwise distances between pseudo CB and ligands        
            Cb_Y_distances = torch.sum((noised_pseudo_cb_coords[:, :, None, :] - Y) ** 2, -1)            
            mask_Y = protein_residue_node_mask[:, :, None] * Y_m 
            Cb_Y_distances_adjusted = Cb_Y_distances * mask_Y + (1.0 - mask_Y) * 10000.0
            _, E_idx_Y = torch.topk(
                Cb_Y_distances_adjusted, self.ligand_atom_context_num, dim=-1, largest=False
            ) # E_idx_Y is the indices of the ligand atoms that are closest to each of the pseudo CBs
            
            # Gather Y, Y_t, Y_m that are closest to each of the pseudo CBs
            Y = torch.gather(Y, 2, E_idx_Y[:, :, :, None].repeat(1, 1, 1, 3))
            Y_t = torch.gather(Y_t, 2, E_idx_Y)
            Y_m = torch.gather(Y_m, 2, E_idx_Y)                        
                
            # Atom type information for ligand & sidechain atoms # Todo: handle Lanthanide metals properly            
            Y_t = Y_t.long()
            Y_t_g = torch.tensor(PERIODIC_TABLE_FEATURES[1], device=Y_t.device)[Y_t]  # group; 19 categories including 0
            Y_t_p = torch.tensor(PERIODIC_TABLE_FEATURES[2], device=Y_t.device)[Y_t]  # period; 8 categories including 0
            Y_t_g_1hot_ = torch.nn.functional.one_hot(Y_t_g, 19)  # [B, L, M, 19]
            Y_t_p_1hot_ = torch.nn.functional.one_hot(Y_t_p, 8)  # [B, L, M, 8]
            Y_t_1hot_ = torch.nn.functional.one_hot(Y_t, 120)  # [B, L, M, 120]
            Y_t_1hot_ = torch.cat([Y_t_1hot_, Y_t_g_1hot_, Y_t_p_1hot_], -1)  # [B, L, M, 147]
            Y_t_1hot = self.type_linear(Y_t_1hot_.float())
            
            # Generate RBF features for backbone (+ pseudo CB) and ligands      
            if self.protein_ligand_interaction_rbf_type == "cb":
                D_ligand_to_backbone_or_pseudocb = torch.sqrt(
                torch.sum(
                    (Y[:, :, :, None, :]
                     - noised_backbone_pseudo_cb_coords[:, :, 4:, :][:, :, None, :, :]) 
                    ** 2,
                    dim=-1,
                )
                + 1e-6
            )         
            else:
                D_ligand_to_backbone_or_pseudocb = torch.sqrt(
                torch.sum(
                    (Y[:, :, :, None, :]
                     - noised_backbone_pseudo_cb_coords[:, :, None, :, :]) 
                    ** 2,
                    dim=-1,
                )
                + 1e-6
            )    
                
                    
            RBF_ligand_to_backbone_or_pseudocb = self.compute_rbf_embedding_from_distances(D = D_ligand_to_backbone_or_pseudocb)
            RBF_ligand_to_backbone_or_pseudocb = RBF_ligand_to_backbone_or_pseudocb.view(RBF_ligand_to_backbone_or_pseudocb.shape[0], RBF_ligand_to_backbone_or_pseudocb.shape[1], RBF_ligand_to_backbone_or_pseudocb.shape[2], -1)                                    
                                
            # Make angle features between backbone and ligand atoms
            if self.add_angle_features:
                angle_features = self._make_angle_features(noised_backbone_pseudo_cb_coords[:, :, 0, :], noised_backbone_pseudo_cb_coords[:, :, 1, :], noised_backbone_pseudo_cb_coords[:, :, 2, :], Y) # N, Ca, C / # [B, L, M, 4]                

            # Make ligand-protein interaction features by concatenating RBF features, ligand atom type information, and angle features.
            #! (JH) Not sure why Y_t_1hot is concatenated here, maybe let the model know each of the atom types so that the model learn the "interaction" between different types of atoms?
            if self.add_angle_features:
                D_all = torch.cat((RBF_ligand_to_backbone_or_pseudocb, Y_t_1hot, angle_features), dim=-1) # [B, L, M, num_bins + 64 + 4] or [B, L, M, 5 * num_bins + 64 + 4]
            else:                
                D_all = torch.cat((RBF_ligand_to_backbone_or_pseudocb, Y_t_1hot), dim=-1)  # [B,L,M,num_bins+64] or [B,L,M,5*num_bins+64] 
            V = self.node_project_down(D_all)  # [B, L, M, node_features]
            V = self.norm_nodes(V) 

            ###### Ligand subgraph features ######
            # ligand subgraph nodes
            Y_nodes = self.y_nodes(Y_t_1hot_.float())
            Y_nodes = self.norm_y_nodes(Y_nodes)
            
            # ligand subgraph edges (pairwise distances between ligand atoms and sidechain atoms)            
            Y_edges = self._rbf(
                torch.sqrt(
                    torch.sum((Y[:, :, :, None, :] - Y[:, :, None, :, :]) ** 2, -1) + 1e-6
                )
            )  # [B, L, M, M, num_bins]
            Y_edges = self.y_edges(Y_edges)            
            Y_edges = self.norm_y_edges(Y_edges)                        
        else:
            V = None
            Y_nodes = None
            Y_edges = None
            Y_m = None
            
        return E, E_idx, V, Y_nodes, Y_edges, Y_m, D_neighbors 


    def _get_protein_token_center_coords(self, batch: dict[str, TensorType["b ..."]]) -> TensorType["b n 3", float]:
        """
        Get protein token-level center coordinates. Standard amino acid only.
        """
        B, N, _ = batch["noised_coords"].shape
        X = batch["noised_coords"][torch.arange(B).unsqueeze(-1), batch["token_to_center_atom"]]  # get center atom for each token, ca for proteins                
        X = X * batch["protein_residue_node_mask"].unsqueeze(-1)
        return X

    def _get_token_coords(self, batch: dict[str, TensorType["b ..."]], protein_only: bool = True) -> TensorType["b n 3", float]:
        """
        Get token-level coordinates as an average over all known, resolved atoms in the token.
        """
        B, N, _ = batch["coords"].shape
        X = batch["coords"][torch.arange(B).unsqueeze(-1), batch["token_to_center_atom"]]  # get center atom for each token        
        if protein_only:
            X = X * batch["token_is_protein_chain"].unsqueeze(-1)
        X = X * batch["token_exists_mask"].unsqueeze(-1)  # mask out padding and unresolved atoms
        return X        
                
    def _dist(self, X = None, mask = None, eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.k_neighbors, X.shape[1]), dim=-1, sorted=True, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = self.min_rbf_mean, self.max_rbf_mean, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B
    
    def get_backbone_pseudocb_rbf(self, batch: dict[str, TensorType["b ..."]] = None,
                            D_neighbors = None,
                            E_idx = None,
                            rbf_type = "ncacocb") -> TensorType["b n_tokens n_tokens num_rbf", float]:
        
        ca_coords = batch["noised_ca_coords"]
        n_coords = batch["noised_n_coords"]
        c_coords = batch["noised_c_coords"]
        o_coords = batch["noised_o_coords"]
        pseudo_cb_coords = batch["noised_pseudo_cb_coords"]
        
        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(n_coords, n_coords, E_idx))  # N-N
        RBF_all.append(self._get_rbf(c_coords, c_coords, E_idx))  # C-C
        RBF_all.append(self._get_rbf(o_coords, o_coords, E_idx))  # O-O
        if rbf_type == "ncacocb":
            RBF_all.append(self._get_rbf(pseudo_cb_coords, pseudo_cb_coords, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(ca_coords, n_coords, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(ca_coords, c_coords, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(ca_coords, o_coords, E_idx))  # Ca-O
        if rbf_type == "ncacocb":
            RBF_all.append(self._get_rbf(ca_coords, pseudo_cb_coords, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(n_coords, c_coords, E_idx))  # N-C
        RBF_all.append(self._get_rbf(n_coords, o_coords, E_idx))  # N-O
        if rbf_type == "ncacocb":
            RBF_all.append(self._get_rbf(n_coords, pseudo_cb_coords, E_idx))  # N-Cb
            RBF_all.append(self._get_rbf(pseudo_cb_coords, c_coords, E_idx))  # Cb-C
            RBF_all.append(self._get_rbf(pseudo_cb_coords, o_coords, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(o_coords, c_coords, E_idx))  # O-C
        RBF_all.append(self._get_rbf(n_coords, ca_coords, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(c_coords, ca_coords, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(o_coords, ca_coords, E_idx))  # O-Ca
        if rbf_type == "ncacocb":
            RBF_all.append(self._get_rbf(pseudo_cb_coords, ca_coords, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(c_coords, n_coords, E_idx))  # C-N
        RBF_all.append(self._get_rbf(o_coords, n_coords, E_idx))  # O-N
        if rbf_type == "ncacocb":
            RBF_all.append(self._get_rbf(pseudo_cb_coords, n_coords, E_idx))  # Cb-N
            RBF_all.append(self._get_rbf(c_coords, pseudo_cb_coords, E_idx))  # C-Cb
            RBF_all.append(self._get_rbf(o_coords, pseudo_cb_coords, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(c_coords, o_coords, E_idx))  # C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)
        
        return RBF_all
        
    def compute_rbf_embedding_from_distances(self, D = None):
        """
        Given a tensor of pairwise distances, compute the radial basis
        embedding of the distances.

        Args:
            D (torch.Tensor): [B, L, M] or [B, L, M, N] - Pairwise distances between each
                residue's representative atom, masked by the 2D mask.
        Returns:
            rbf_embedding (torch.Tensor): [B, L, M, num_rbf] or [B, L, M, N, num_rbf] - Radial basis
                function embedding of the pairwise distances.
        """
        # Linear space the means of the radial basis functions.
        
        rbf_mus = torch.linspace(
            self.min_rbf_mean, self.max_rbf_mean, self.num_rbf, device=D.device
        )
        
        if len(D.shape) == 3: 
            rbf_mus = rbf_mus[None, None, None, :]
        elif len(D.shape) == 4:
            rbf_mus = rbf_mus[None, None, None, None, :]        

        # The standard deviation of the radial basis functions.
        rbf_sigma = (self.max_rbf_mean - self.min_rbf_mean) / self.num_rbf

        # Expand the dimensions of D to match the shape of rbf_mus.
        # D_expand: [B, L, M, 1] or [B, L, M, N, 1]
        D_expand = torch.unsqueeze(D, -1)

        # Compute the radial basis function embedding.
        # RBF: [B, L, M, num_rbf] or [B, L, M, N, num_rbf]
        rbf_embedding = torch.exp(-(((D_expand - rbf_mus) / rbf_sigma) ** 2))

        return rbf_embedding
                                                                     
    # def _get_rbf(self, A, B, E_idx): #! Assuming A and B are the coordinates of the single type atom (e.g. CA or CB)
    #     """
    #     (JH) Memory efficient version of _get_rbf. O(N^2) -> O(NK).
    #     """
                
    #     K, C = E_idx.size(-1), A.size(-1)
        
    #     B_neighbors = torch.gather(
    #         B.unsqueeze(2).expand(-1, -1, K, -1),
    #         dim=1,
    #         index=E_idx.unsqueeze(-1).expand(-1, -1, -1, C)
    #     )
        
    #     diff = A.unsqueeze(2) - B_neighbors
    #     D = torch.sqrt((diff*diff).sum(-1) + 1e-6)
    #     RBF_A_B = self._rbf(D)
            
    #     return RBF_A_B                
    
    def _make_angle_features(self, A, B, C, Y): #! from ligandMPNN
        v1 = A - B
        v2 = C - B
        e1 = torch.nn.functional.normalize(v1, dim=-1)
        e1_v2_dot = torch.einsum("bli, bli -> bl", e1, v2)[..., None]
        u2 = v2 - e1 * e1_v2_dot
        e2 = torch.nn.functional.normalize(u2, dim=-1)
        e3 = torch.cross(e1, e2, dim=-1)
        R_residue = torch.cat(
            (e1[:, :, :, None], e2[:, :, :, None], e3[:, :, :, None]), dim=-1
        )

        local_vectors = torch.einsum(
            "blqp, blyq -> blyp", R_residue, Y - B[:, :, None, :]
        )

        rxy = torch.sqrt(local_vectors[..., 0] ** 2 + local_vectors[..., 1] ** 2 + 1e-8)
        f1 = local_vectors[..., 0] / rxy
        f2 = local_vectors[..., 1] / rxy
        rxyz = torch.norm(local_vectors, dim=-1) + 1e-8
        f3 = rxy / rxyz
        f4 = local_vectors[..., 2] / rxyz

        f = torch.cat([f1[..., None], f2[..., None], f3[..., None], f4[..., None]], -1)
        return f
    
    def _get_nearest_ligand_atoms(self, CB = None, 
                                  mask = None, 
                                  Y = None, 
                                  Y_t = None, 
                                  Y_m = None, 
                                  number_of_ligand_atoms = 16, 
                                  device = None):            
        
        """
        batchfied version of _get_nearest_neighbours in data_utils.py of LigandMPNN.
        """
        
        mask_CBY = mask[:, :, None] * Y_m[:, None, :]  # [A,B]
        L2_AB = torch.sum((CB[:, :, None, :] - Y[:, None, :, :]) ** 2, -1)
        L2_AB = L2_AB * mask_CBY + (1 - mask_CBY) * 1000.0

        nn_idx = torch.argsort(L2_AB, -1)[:, :, :number_of_ligand_atoms]
        L2_AB_nn = torch.gather(L2_AB, -1, nn_idx)
        D_AB_closest = torch.sqrt(L2_AB_nn[:, :, 0])

        Y_r = Y.unsqueeze(1).repeat(1, CB.shape[1], 1, 1)
        Y_t_r = Y_t.unsqueeze(1).repeat(1, CB.shape[1], 1)
        Y_m_r = Y_m.unsqueeze(1).repeat(1, CB.shape[1], 1)

        # Y_r = Y[None, :, :].repeat(CB.shape[0], 1, 1)
        # Y_t_r = Y_t[None, :].repeat(CB.shape[0], 1)
        # Y_m_r = Y_m[None, :].repeat(CB.shape[0], 1)
                
        Y_tmp = torch.gather(Y_r, 2, nn_idx[:, :, :, None].repeat(1, 1, 1, 3))
        Y_t_tmp = torch.gather(Y_t_r, 2, nn_idx)
        Y_m_tmp = torch.gather(Y_m_r, 2, nn_idx)

        Y = torch.zeros(
            [CB.shape[0], CB.shape[1], number_of_ligand_atoms, 3], dtype=torch.float32, device=device
        )
        Y_t = torch.zeros(
            [CB.shape[0], CB.shape[1], number_of_ligand_atoms], dtype=torch.int32, device=device
        )
        Y_m = torch.zeros(
            [CB.shape[0], CB.shape[1], number_of_ligand_atoms], dtype=torch.int32, device=device
        )

        num_nn_update = Y_tmp.shape[2]
        Y[:, :, :num_nn_update] = Y_tmp
        Y_t[:, :, :num_nn_update] = Y_t_tmp
        Y_m[:, :, :num_nn_update] = Y_m_tmp

        return Y, Y_t, Y_m, D_AB_closest



class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = torch.nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = torch.nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class PositionalEncodings(torch.nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = torch.nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E

class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.num_hidden)
        self.norm2 = nn.LayerNorm(self.num_hidden)

        self.W1 = nn.Linear(self.num_hidden + num_in, self.num_hidden, bias=True)
        self.W2 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
        self.W3 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden * 2 + num_in, num_hidden, bias=True) # nh * 2 for vi AND vj
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_in, bias=True) # num_in is hidden dim of edges h_E
        self.norm3 = nn.LayerNorm(num_in)
        self.dropout3 = nn.Dropout(dropout)

        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(self.num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, E_idx = None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message

        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        #edge updates        
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        
        # if mask_attend is not None: #! (JH) fixed 251009
        #     h_E = mask_attend.unsqueeze(-1) * h_E

        return h_V, h_E

class Contextfeatureprocessor(nn.Module): # self.y_context_encoder_layers in ligandMPNN
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, 
                 scale=30, edge_update=False):
        super(Contextfeatureprocessor, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.num_hidden)
        self.norm2 = nn.LayerNorm(self.num_hidden)

        self.W1 = nn.Linear(self.num_hidden * 2 + self.num_in, self.num_hidden, bias=True) # Following the foundry's LigandMPNN implementation
        self.W2 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
        self.W3 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
        self.W11 = nn.Linear(self.num_hidden * 2 + self.num_in, self.num_hidden, bias=True) # nh * 2 for vi AND vj
        self.W12 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
        self.W13 = nn.Linear(self.num_hidden, self.num_in, bias=True) # num_in is hidden dim of edges h_E
        self.norm3 = nn.LayerNorm(self.num_in)
        self.dropout3 = nn.Dropout(dropout)
    
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(self.num_hidden, num_hidden * 4)
        
        self.edge_update = edge_update

    # @dynamo.disable()
    def forward(self, h_V = None, h_E = None, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Source node features
        h_V_i = h_V.unsqueeze(-2).expand(-1,-1,-1,h_E.size(-2),-1) # [B, L, M, M, C_node]
        
        # Destination node features
        h_V_j = h_V.unsqueeze(-3).expand(-1, -1, h_E.size(-3), -1, -1)  # [B, L, M, M, C_node]

        h_EV = torch.cat([h_V_i, h_E, h_V_j], -1) # [B, L, M, M, C_edge + C_node + C_node]        
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message

        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V        
            
        if self.edge_update: #Todo: fix
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,-1,h_EV.size(-2),-1)
            h_EV = torch.cat([h_V_expand, h_EV], -1)
            h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
            h_E = self.norm3(h_E + self.dropout3(h_message))
            
            if mask_attend is not None: #! (JH) fixed 251009
                h_E = mask_attend.unsqueeze(-1) * h_E
                
            return h_V, h_E
        else:
            return h_V, None    

class Contextfeatureaggregator(nn.Module): #! (JH) self.context_encoder_layers in ligandMPNN
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, 
                 scale=30, edge_update=False):
        super(Contextfeatureaggregator, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.num_hidden)
        self.norm2 = nn.LayerNorm(self.num_hidden)

        self.W1 = nn.Linear(self.num_hidden + num_in, self.num_hidden, bias=True)
        self.W2 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
        self.W3 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
        self.W11 = nn.Linear(self.num_hidden + self.num_in, self.num_hidden, bias=True) # self.num_hidden for node features and self.num_in for edge features
        self.W12 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
        self.W13 = nn.Linear(self.num_hidden, self.num_in, bias=True) # num_in is hidden dim of edges h_E
        self.norm3 = nn.LayerNorm(self.num_in)
        self.dropout3 = nn.Dropout(dropout)

        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(self.num_hidden, num_hidden * 4)
        
        self.edge_update = edge_update

    # @dynamo.disable()
    def forward(self, h_V = None, h_E = None, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        #! h_message here is the context features for each protein node

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message

        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        # edge updates
        if self.edge_update: # Todo: fix
            h_EV = torch.cat([h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1), h_E], dim=-1)
            #! (JH) already Y_nodes are concatenated to h_E
            h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
            h_E = self.norm3(h_E + self.dropout3(h_message))
            
            if mask_attend is not None: #! (JH) fixed 251009
                h_E = mask_attend.unsqueeze(-1) * h_E
                
            return h_V, h_E
        
        else:
            return h_V, None

class ContextModule(nn.Module):
    def __init__(self, hidden_dim: int, dropout_p: float, num_processor_layers: int, num_aggregator_layers: int, edge_update: bool):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_update = edge_update

        # Projections
        self.W_v = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.W_c = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.W_nodes_y = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.W_edges_y = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.V_C = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.V_C_norm = torch.nn.LayerNorm(self.hidden_dim)
        self.dropout = torch.nn.Dropout(dropout_p)

        # Stacks
        self.context_feature_processor = torch.nn.ModuleList(
            [Contextfeatureprocessor(self.hidden_dim, self.hidden_dim, dropout=dropout_p, edge_update=self.edge_update) for _ in range(num_processor_layers)]
        )
        self.context_feature_aggregator = torch.nn.ModuleList(
            [Contextfeatureaggregator(self.hidden_dim, self.hidden_dim * 2, dropout=dropout_p, edge_update=self.edge_update) for _ in range(num_aggregator_layers)]
        )

    # @dynamo.disable()
    def forward(self, h_V = None, h_E = None, 
                V = None, Y_nodes = None, Y_edges = None, 
                Y_m = None, E_idx = None,
                protein_residue_node_mask = None):
        # Guard: if no context, return h_V unchanged
        if V is None or Y_nodes is None or Y_edges is None or Y_m is None:
            return h_V

        h_E_context = self.W_v(V)
        h_V_C = self.W_c(h_V)
        Y_m_edges = Y_m[:, :, :, None] * Y_m[:, :, None, :]
        Y_nodes = self.W_nodes_y(Y_nodes)
        Y_edges = self.W_edges_y(Y_edges)

        if not self.edge_update:
            for i in range(len(self.context_feature_aggregator)):
                Y_nodes, _ = self.context_feature_processor[i](
                    h_V=Y_nodes, h_E=Y_edges, mask_V=Y_m, mask_attend=Y_m_edges,
                )
                h_E_context_cat = torch.cat([h_E_context, Y_nodes], -1)
                h_V_C, _ = self.context_feature_aggregator[i](
                    h_V=h_V_C, h_E=h_E_context_cat, mask_V=protein_residue_node_mask, mask_attend=Y_m
                )
        else: #Todo: fix
            h_E_context_cat = torch.cat([h_E_context, Y_nodes], -1)
            for i in range(len(self.context_feature_aggregator)):
                Y_nodes, Y_edges = self.context_feature_processor[i](
                    h_V=Y_nodes, h_E=Y_edges, mask_V=Y_m, mask_attend=Y_m_edges, E_idx=E_idx_YY
                )
                h_V_C, h_E_context_cat = self.context_feature_aggregator[i](
                    h_V=h_V_C, h_E=h_E_context_cat, mask_V=prot_token_mask, mask_attend=Y_m
                )

        h_V_C = self.V_C(h_V_C)
        h_V = h_V + self.V_C_norm(self.dropout(h_V_C))
        return h_V

class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30, is_last_layer=False):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.is_last_layer = is_last_layer

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

        if not is_last_layer:
            # only initialize if not last layer to avoid unused parameters
            self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
            self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
            self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
            self.norm3 = nn.LayerNorm(num_hidden)


    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        
        # Edge updates
        if not self.is_last_layer:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
            h_EV = torch.cat([h_V_expand, h_EV], -1)
            h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
            h_E = self.norm3(h_E + self.dropout3(h_message))
        
        # if mask_attend is not None: #! (JH) fixed 251009
        #     h_E = mask_attend.unsqueeze(-1) * h_E
                
        return h_V, h_E


def get_tokenwise_coords(coords: TensorType["b n_atoms 3", float],
                         tokenwise_atom_idxs: TensorType["b n_tokens"],
                         tokenwise_atom_idxs_mask: TensorType["b n_tokens"],
                         ) -> TensorType["b n_tokens MAX_NUM_ATOMS 3", float]:
    """
    Get token-level coordinates (padded to max_num_atoms per token). Batched version of pad_atom_feats_to_tokenwise for just coords.
    tokenwise_atom_idxs_mask is basically token_pad_mask for MAX_NUM_ATOMS atoms per token.
    """
    
    tokenwise_coords = batched_gather(coords, tokenwise_atom_idxs, dim=1, no_batch_dims=1) * tokenwise_atom_idxs_mask[..., None]

    return tokenwise_coords
    
def get_atomwise_coords(
    batch: dict[str, TensorType["b ..."]],
    tokenwise_coords: TensorType["b n_tokens 23 3", float],
) -> TensorType["b n_atoms 3", float]:
    """
    Inverse of get_tokenwise_coords. Given tokenwise coords [B, n_tokens, max_num_atoms, 3],
    reconstruct atomwise coords [B, n_atoms, 3].
    """
    B = batch["coords"].shape[0]
    device = batch["coords"].device

    x = batch["atomwise_token_idx"] * tokenwise_coords.shape[-2]  # flattened atomwise token indices
    is_start = torch.ones_like(x, dtype=torch.bool)
    is_start[:, 1:] = x[:, 1:] != x[:, :-1]
    pos = torch.arange(x.shape[-1], device=x.device).unsqueeze(0).expand(B, x.shape[-1])
    start_pos = torch.where(is_start, pos, torch.full_like(pos, -1))
    first_pos = torch.cummax(start_pos, dim=1).values
    local_idx = pos - first_pos
    gather_idx = x + local_idx
    gather_idx = gather_idx

    new_coords = batched_gather(tokenwise_coords.view(B, -1, 3), gather_idx, dim=1, no_batch_dims=1)
    return new_coords
