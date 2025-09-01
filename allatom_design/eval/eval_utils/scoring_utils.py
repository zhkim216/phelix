from typing import Any, Callable, Dict, Optional, Tuple, List, Union
import numpy as np
import torch
from torchtyping import TensorType
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch
import numpy as np
from allatom_design.data import residue_constants as rc
from einops import rearrange
from functools import partial
from scipy.stats import pearsonr, spearmanr
from omegaconf import DictConfig
import yaml
import os

def apply_mutation_batched(x: TensorType["b n a 3", float],
                   aatype: TensorType["b n", int],
                   seq_mask: TensorType["b n", int],
                   missing_atom_mask: TensorType["b n a", int],
                   mutations: list[list[str]]
                   ) -> tuple[TensorType["b n a 3", float],
                              TensorType["b n", int],
                              TensorType["b n", int]]:

    b = aatype.shape[0]
    seq_mlm_mask = seq_mask.clone()
    mut_positions = np.zeros(b)
    mut_res_idxs = np.zeros(b)
    wt_res_idxs = np.zeros(b)
    wt_example_mask = np.full(b, False)

    # Loop through the batch and apply mutations
    for i in range(b):

        #parse mutation info
        mut = mutations[i]

        if mut == 'wt':
            wt_example_mask[i] = True
            continue

        wt_res, pos, mut_res = mut[0], int(mut[1:-1]), mut[-1]

        #save positions and wt residue indices for later evaluation
        mut_positions[i] = pos
        mut_res_idxs[i] = rc.restype_order_with_x[mut_res]
        wt_res_idxs[i] = rc.restype_order_with_x[wt_res]

        #error handling
        if aatype[i, pos] != rc.restype_order_with_x[wt_res]:
            with open('error.txt', 'w') as fh:
                fh.write(''.join([rc.idx_to_restype_with_x[int(res)] for res in aatype[i,:]]))

        assert aatype[i, pos] == rc.restype_order_with_x[wt_res], f'Mutation info and sequence do not match!! {[rc.idx_to_restype_with_x[int(res)] for res in aatype[i,pos-5:pos+5]]} {wt_res}{pos}{mut_res}'

        # Update the mask (set to 0 where mutation is applied)
        seq_mlm_mask[i, pos] = 0

    # Mask the sidechains at the mutated positions in x
    x_masked = x.clone()
    x_masked[..., rc.non_bb_idxs, :] = x[..., rc.non_bb_idxs, :] * rearrange(seq_mlm_mask, "b n -> b n 1 1").float()
    missing_atom_mask[:, :, rc.non_bb_idxs] *= rearrange(seq_mlm_mask, "b n -> b n 1").float()

    # Mask the amino acid type
    aatype_masked = aatype.clone()
    aatype_masked = torch.where(seq_mlm_mask.bool(), aatype, rc.restype_order_with_x["X"]) #is this okay? it sets padded positions to X as well lol
    aatype_masked = torch.where(seq_mask.bool(), aatype_masked, 0) #fixed this^ by repadding with 0, which is also weird cause thats alanine lol

    return x_masked, seq_mlm_mask, missing_atom_mask, aatype_masked, mut_positions, mut_res_idxs, wt_res_idxs, wt_example_mask

def mutate_whole_seq(model,
                     x: TensorType["b n a 3", float],
                     aatype: TensorType["b n", int],
                     mutation_list: list[list[str]],
                     seq_mask: TensorType["b n", int],
                     residue_index: TensorType["b n", int],
                     missing_atom_mask: TensorType["b n a", float],
                     chain_index: TensorType["b n", int],
                     scd_inputs: Dict,
                     max_number_muts: int,
                    ) -> tuple[TensorType["b n a 3", float],
                              TensorType["b n", int]]:
    aatype_mut = aatype.clone()
    x_mut = x.clone()
    b = x.shape[0]
    mut_positions = np.full((b, max_number_muts), 0)
    mut_res_idxs = np.full((b, max_number_muts), 0)
    wt_res_idxs = np.full((b, max_number_muts), 0)
    padded_mutations_mask = torch.full((b, max_number_muts), True)
    wt_example_mask = np.full(b, False)

    aux_inputs = {"scd": scd_inputs,
                  "seq_mlm_mask": seq_mask.clone(),
                  "scn_mlm_mask": seq_mask.clone()}


    #mutation_list is nested list as there may be more than one mutation per sequence
    for i in range(b):
        mutations = mutation_list[i]
        for mut_num, mut in enumerate(mutations.split(':')):

            if mut =='wt':
                wt_example_mask[i] = True
                continue

            wt_res, pos, mut_res = mut[0], int(mut[1:-1]), mut[-1]

            assert aatype[i, pos] == rc.restype_order_with_x[wt_res], f'Mutation info and sequence do not match!! {[rc.idx_to_restype_with_x[int(res)] for res in aatype[i,pos-5:pos+5]]} {wt_res}{pos}{mut_res}'

            #apply mutation to sequence
            aatype_mut[i, pos] = rc.restype_order_with_x[mut_res]

            #update scn mask to later mask out wt sidechain coords at mutant position
            aux_inputs['scn_mlm_mask'][i, pos] = 0

            #store info
            mut_positions[i, mut_num] = pos
            padded_mutations_mask[i, mut_num] = False
            mut_res_idxs[i, mut_num] = rc.restype_order_with_x[mut_res]
            wt_res_idxs[i, mut_num] = rc.restype_order_with_x[wt_res]

    #teacher force mutated sequence
    aux_inputs['scd']["aatype_override"] = aatype_mut
    aux_inputs['scd']["aatype_override_mask"] = seq_mask.clone()

    #zero out sidechains of mutated residues we want to pack
    x_mut[:, :, rc.non_bb_idxs, :] = x[:, :, rc.non_bb_idxs, :]  * rearrange(aux_inputs['scn_mlm_mask'], "b n -> b n 1 1").float()
    missing_atom_mask[:, :, rc.non_bb_idxs] *= rearrange(aux_inputs['scn_mlm_mask'], "b n -> b n 1").float()

    #run model
    x1_mut_pred, _, _ = model.model.denoiser(x_mut,
                                            aatype_mut,
                                            residue_index=residue_index,
                                            seq_mask=seq_mask,
                                            chain_encoding=chain_index,
                                            missing_atom_mask=missing_atom_mask,
                                            scn_mlm_mask = aux_inputs['scn_mlm_mask'],
                                            aux_inputs=aux_inputs,
                                            t=torch.ones_like(seq_mask),
                                            is_sampling=True
                                        )

    #update structure with newly packed residues
    x_mut[torch.arange(b).unsqueeze(1), mut_positions,...] = x1_mut_pred[torch.arange(b).unsqueeze(1), mut_positions,...]

    return aatype_mut, x_mut, mut_positions, mut_res_idxs, wt_res_idxs, wt_example_mask, padded_mutations_mask

def score_seq(model,
              x: TensorType["b n a 3", float],
              aatype: TensorType["b n", int],
              seq_mask: TensorType["b n", int],
              residue_index: TensorType["b n", int],
              missing_atom_mask: TensorType["b n a", float],
              chain_index: TensorType["b n", int],
              mutations: list[list[str]],
              scd_inputs: Dict,
              method: str
            ) -> TensorType["b n", int]:

    B = x.shape[0]

    if method == 'single':
        x_masked, seq_mlm_mask, missing_atom_mask, aatype_masked, mut_positions, mut_res_idxs, wt_res_idxs, wt_example_mask = apply_mutation_batched(x, aatype, seq_mask, missing_atom_mask, mutations)

        #score examples
        logits = model.model.score(
            x_masked,
            aatype_masked,
            missing_atom_mask=missing_atom_mask,
            seq_mask=seq_mask,
            residue_index=residue_index,
            chain_index=chain_index,
        )

        scores = logits[torch.arange(len(logits)), mut_positions, mut_res_idxs] - logits[torch.arange(len(logits)), mut_positions, wt_res_idxs]

        #score for wt examples should be 0
        scores[wt_example_mask] = 0

    elif method == 'multiple':

        #we will eval O(max_number_of_muts)  so we can parallelize
        max_number_muts = max([len(muts.split(':')) for muts in mutations])
        scores = torch.zeros((B, max_number_muts))

        aatype_mut, x_mut, mut_positions, mut_res_idxs, wt_res_idxs, wt_example_mask, padded_mutations_mask = mutate_whole_seq(model=model,
                                                                                                        x=x,
                                                                                                        aatype=aatype,
                                                                                                        mutation_list=mutations,
                                                                                                        seq_mask=seq_mask,
                                                                                                        residue_index=residue_index,
                                                                                                        missing_atom_mask=missing_atom_mask,
                                                                                                        chain_index=chain_index,
                                                                                                        scd_inputs=scd_inputs,
                                                                                                        max_number_muts=max_number_muts
                                                                                                        )

        for mut_num in range(max_number_muts):
            #mask seq at position
            seq_mlm_mask = seq_mask.clone()
            seq_mlm_mask[:, mut_positions[:, mut_num]] = 0
            aatype_masked = aatype_mut.clone()
            aatype_masked[:, mut_positions[:, mut_num]] = rc.restype_order_with_x["X"]

            #mask sidechain at position
            x_masked = x.clone()
            x_masked[..., rc.non_bb_idxs, :] = x[..., rc.non_bb_idxs, :] * rearrange(seq_mlm_mask, "b n -> b n 1 1").float()
            missing_atom_mask[..., rc.non_bb_idxs] *=  rearrange(seq_mlm_mask, "b n -> b n 1").float()

            #score examples
            logits = model.model.score(
                x_masked,
                aatype_masked,
                seq_mask=seq_mask,
                missing_atom_mask=missing_atom_mask,
                residue_index=residue_index,
                chain_index=chain_index,
            )

            scores[:, mut_num] = logits[torch.arange(len(logits)), mut_positions[:, mut_num], mut_res_idxs[:, mut_num]] - logits[torch.arange(len(logits)), mut_positions[:, mut_num], wt_res_idxs[:, mut_num]]

        #ignore dummy mutation position
        scores = torch.where(padded_mutations_mask, 0, scores)
        scores = torch.sum(scores, dim = -1)

        #score for wt examples should be 0
        scores[wt_example_mask] = 0

    elif method == 'pseudo_ppl':
        b = x.shape[0]

        #more efficient so we don't iterate over padded positions
        max_length = int(max(torch.sum(seq_mask, dim = -1)))
        pseudo_ppl_logits = torch.zeros((b, max_length, rc.restype_with_x_num), device = x.device)
        mut_pseudo_ppl_logits = pseudo_ppl_logits.clone()
        aatype_mut, x_mut, _, _, _, wt_example_mask, _ = mutate_whole_seq(model=model,
                                             x=x,
                                             aatype=aatype,
                                             mutation_list=mutations,
                                             seq_mask=seq_mask,
                                             missing_atom_mask=missing_atom_mask,
                                             residue_index=residue_index,
                                             chain_index=chain_index,
                                             scd_inputs=scd_inputs)

        for i in range(max_length):

            #apply mask to mlm_mask
            seq_mlm_mask = seq_mask.clone()
            seq_mlm_mask[:, i] = 0

            #mask sidechain according to mlm_mask, for wt and mut
            x_masked = x.clone()
            x_masked[..., rc.non_bb_idxs, :] = x[..., rc.non_bb_idxs, :] * rearrange(seq_mlm_mask, "b n -> b n 1 1").float()

            x_mut_masked = x_mut.clone()
            x_mut_masked[..., rc.non_bb_idxs, :] = x_mut[..., rc.non_bb_idxs, :] * rearrange(seq_mlm_mask, "b n -> b n 1 1").float()

            #mask sequence accoriding to mlm_mask, for wt and mut
            aatype_masked = aatype.clone()
            aatype_masked = torch.where(seq_mlm_mask.bool(), aatype, rc.restype_order_with_x["X"])

            aatype_mut_masked = aatype_mut.clone()
            aatype_mut_masked = torch.where(seq_mlm_mask.bool(), aatype_mut, rc.restype_order_with_x["X"])

            #crop everything to max_length for effiiency
            logits = model.model.score(
                x_masked[:, :max_length, ...],
                aatype_masked[:, :max_length],
                seq_mask=seq_mask[:, :max_length],
                residue_index=residue_index[:, :max_length],
                chain_index=chain_index[:, :max_length],
            )

            mut_logits = model.model.score(
                x_mut_masked[:, :max_length, ...],
                aatype_mut_masked[:, :max_length],
                seq_mask=seq_mask[:, :max_length],
                residue_index=residue_index[:, :max_length],
                chain_index=chain_index[:, :max_length],
            )

            pseudo_ppl_logits[:, i, :] = logits[:, i, :]
            mut_pseudo_ppl_logits[:, i, :] = mut_logits[:, i, :]

        #needed for indexing
        batch_indices = torch.arange(pseudo_ppl_logits.shape[0], device = x.device)[:, None]
        seq_indices = torch.arange(pseudo_ppl_logits.shape[1], device = x.device)

        #get wt psuedo_ppl
        wt_pseudo_ppl = pseudo_ppl_logits[batch_indices, seq_indices, aatype[:, :max_length]]
        wt_pseudo_ppl *= seq_mask[:, :max_length] #zero out padded positions
        wt_pseudo_ppl = torch.sum(wt_pseudo_ppl, dim = -1) #sum over residue dimension

        #get mut pseudo_ppl
        mut_pseudo_ppl = mut_pseudo_ppl_logits[batch_indices, seq_indices, aatype_mut[:, :max_length]]
        mut_pseudo_ppl *= seq_mask[:, :max_length] #zero out padded positions
        mut_pseudo_ppl = torch.sum(mut_pseudo_ppl, dim = -1) #sum over residue dimension

        #normalize mutant scores by wild-type
        scores = mut_pseudo_ppl - wt_pseudo_ppl

        #score for wt examples should be 0
        scores[wt_example_mask] = 0

    else:
        raise ValueError(f'Incorrect scoring method given: {method}, choose between: single, pseudo_ppl')

    return scores


def get_avg_metrics(scores_exp: Dict,
                    labels_exp: Dict
                    ):

    experiments = scores_exp.keys()
    return np.mean([pearsonr(scores_exp[exp], labels_exp[exp]).correlation for exp in experiments]), np.mean([spearmanr(scores_exp[exp], labels_exp[exp]).correlation for exp in experiments])


def update_data_cfg(data_cfg: DictConfig
                    ):

    dataset_path = data_cfg.pdb_path
    with open(os.path.join(dataset_path,'config.yaml'), 'r') as file:
        dataset_config = yaml.safe_load(file)

    for k, v in dataset_config.items():
        data_cfg[k] = v

    return data_cfg