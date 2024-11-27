from typing import Dict, List
from torchtyping import TensorType
import torch
from allatom_design.data import residue_constants as rc
from einops import rearrange
from allatom_design.eval import scoring_utils

def score_seq_multichain(model, 
              x: TensorType["b n a 3", float],               
              aatype: TensorType["b n", int], 
              seq_mask: TensorType["b n", int], 
              residue_index: TensorType["b n", int],
              chain_index: TensorType["b n", int],
              mutations: List[List[str]],
              scd_inputs: Dict,
              method: str
            ) -> TensorType["b n", int]:
    
    B = x.shape[0]

    if method == 'single':
        x_masked, seq_mlm_mask, aatype_masked, mut_positions, mut_res_idxs, wt_res_idxs, wt_example_mask = scoring_utils.apply_mutation_batched(x, aatype, seq_mask, mutations)

        #score the bound complex
        logits_bound = model.model.score(
            x_masked,
            aatype_masked,
            seq_mlm_mask=seq_mlm_mask,
            seq_mask=seq_mask,
            residue_index=residue_index,
            chain_index=chain_index
        )

        #masks for scoring the unbound chains individually
        chain_1_mask = (chain_index == 0) & (seq_mask == 1) #seq mask needs to be used as well because chain index is padded with 0s
        chain_2_mask = (chain_index == 1) & (seq_mask == 1)

        #evaluate chain 1 unbound by padding out chain 2
        x_masked_chain_1 = torch.where(chain_1_mask[:,None,None], x_masked, 0)
        aatype_masked_chain_1 = torch.where(chain_1_mask, aatype_masked, 0)
        seq_mlm_mask_chain_1 = torch.where(chain_1_mask, seq_mlm_mask, 0)
        seq_mask_chain_1 = torch.where(chain_1_mask, seq_mask, 0)
        residue_index_chain_1 = torch.where(chain_1_mask, residue_index, 0)
        chain_index_chain_1 = torch.where(chain_1_mask, chain_index, 0)

        #score examples for chain 1
        logits_chain_1 = model.model.score(
            x_masked=x_masked_chain_1,
            aatype_masked=aatype_masked_chain_1,
            seq_mlm_mask=seq_mlm_mask_chain_1,
            seq_mask=seq_mask_chain_1,
            residue_index=residue_index_chain_1,
            chain_index=chain_index_chain_1
        )

        #evaluate chain 2 unbound by padding out chain 1
        x_masked_chain_2 = torch.where(chain_2_mask[:,None,None], x_masked, 0)
        aatype_masked_chain_2 = torch.where(chain_2_mask, aatype_masked, 0)
        seq_mlm_mask_chain_2 = torch.where(chain_2_mask, seq_mlm_mask, 0)
        seq_mask_chain_2 = torch.where(chain_2_mask, seq_mask, 0)
        residue_index_chain_2 = torch.where(chain_2_mask, residue_index, 0)
        chain_index_chain_2 = torch.where(chain_2_mask, chain_index, 0)

        #score examples for chain 2
        logits_chain_2 = model.model.score(
            x_masked=x_masked_chain_2,
            aatype_masked=aatype_masked_chain_2,
            seq_mlm_mask=seq_mlm_mask_chain_2,
            seq_mask=seq_mask_chain_2,
            residue_index=residue_index_chain_2,
            chain_index=chain_index_chain_2
        )

        #combine logits of unbound chains
        logits_unbound = torch.where(chain_1_mask, logits_chain_1, 0)
        logits_unbound = torch.where(chain_2_mask, logits_chain_2, logits_unbound)

        #normalize mut scores by unbound likelihoods
        mut_scores = logits_bound[torch.arange(len(logits_bound)), mut_positions, mut_res_idxs] - logits_unbound[torch.arange(len(logits_unbound)), mut_positions, mut_res_idxs]
        
        #normalize wt scores by unbound likelihoods
        wt_scores = logits_bound[torch.arange(len(logits_bound)), mut_positions, wt_res_idxs] - logits_unbound[torch.arange(len(logits_unbound)), mut_positions, wt_res_idxs]
        
        #normalize mut scores by wt scores
        scores = mut_scores - wt_scores

        #score for wt examples should be 0
        scores[wt_example_mask] = 0

    elif method == 'multiple':
        
        #we will eval O(max_number_of_muts)  so we can parallelize
        max_number_muts = max([len(muts.split(':')) for muts in mutations])
        scores = torch.zeros((B, max_number_muts))
        
        aatype_mut, x_mut, mut_positions, mut_res_idxs, wt_res_idxs, wt_example_mask, padded_mutations_mask = scoring_utils.mutate_whole_seq(model=model, 
                                                                                                        x=x, 
                                                                                                        aatype=aatype, 
                                                                                                        mutation_list=mutations, 
                                                                                                        seq_mask=seq_mask, 
                                                                                                        residue_index=residue_index,
                                                                                                        chain_index=chain_index,
                                                                                                        scd_inputs=scd_inputs,
                                                                                                        max_number_muts=max_number_muts
                                                                                                        )

        #masks for scoring the unbound chains individually
        chain_1_mask = (chain_index == 0) & (seq_mask == 1) #seq mask needs to be used as well because chain index is padded with 0s
        chain_2_mask = (chain_index == 1) & (seq_mask == 1)
        
        for mut_num in range(max_number_muts):
            
            #mask seq at position
            seq_mlm_mask = seq_mask.clone() 
            seq_mlm_mask[:, mut_positions[:, mut_num]] = 0
            aatype_masked = aatype_mut.clone()
            aatype_masked[:, mut_positions[:, mut_num]] = rc.restype_order_with_x["X"]

            #mask sidechain at position
            x_masked = x_mut.clone()
            x_masked[..., rc.non_bb_idxs, :] = x_mut[..., rc.non_bb_idxs, :] * rearrange(seq_mlm_mask, "b n -> b n 1 1").float()    
            
            #score the bound complex
            logits_bound = model.model.score(
                x_masked,
                aatype_masked,
                seq_mlm_mask=seq_mlm_mask,
                seq_mask=seq_mask,
                residue_index=residue_index,
                chain_index=chain_index,
            )

            #evaluate chain 1 unbound by padding out chain 2
            x_masked_chain_1 = torch.where(chain_1_mask[:,:,None,None], x_masked, 0)
            aatype_masked_chain_1 = torch.where(chain_1_mask, aatype_masked, 0)
            seq_mlm_mask_chain_1 = torch.where(chain_1_mask, seq_mlm_mask, 0)
            seq_mask_chain_1 = torch.where(chain_1_mask, seq_mask, 0)
            residue_index_chain_1 = torch.where(chain_1_mask, residue_index, 0)
            chain_index_chain_1 = torch.where(chain_1_mask, chain_index, 0)

            #score examples for chain 1
            logits_chain_1 = model.model.score(
                x=x_masked_chain_1,
                aatype=aatype_masked_chain_1,
                seq_mlm_mask=seq_mlm_mask_chain_1,
                seq_mask=seq_mask_chain_1,
                residue_index=residue_index_chain_1,
                chain_index=chain_index_chain_1
            )

            #evaluate chain 2 unbound by padding out chain 1
            x_masked_chain_2 = torch.where(chain_2_mask[:,:,None,None], x_masked, 0)
            aatype_masked_chain_2 = torch.where(chain_2_mask, aatype_masked, 0)
            seq_mlm_mask_chain_2 = torch.where(chain_2_mask, seq_mlm_mask, 0)
            seq_mask_chain_2 = torch.where(chain_2_mask, seq_mask, 0)
            residue_index_chain_2 = torch.where(chain_2_mask, residue_index, 0)
            chain_index_chain_2 = torch.where(chain_2_mask, chain_index, 0)

            #score examples for chain 2
            logits_chain_2 = model.model.score(
                x=x_masked_chain_2,
                aatype=aatype_masked_chain_2,
                seq_mlm_mask=seq_mlm_mask_chain_2,
                seq_mask=seq_mask_chain_2,
                residue_index=residue_index_chain_2,
                chain_index=chain_index_chain_2
            )

            #combine logits of unbound chains
            logits_unbound = torch.where(chain_1_mask[:,:,None], logits_chain_1, 0)
            logits_unbound = torch.where(chain_2_mask[:,:,None], logits_chain_2, logits_unbound)

            #normalize mut scores by unbound likelihoods
            mut_scores = logits_bound[torch.arange(len(logits_bound)), mut_positions[:, mut_num], mut_res_idxs[:, mut_num]] - logits_unbound[torch.arange(len(logits_unbound)), mut_positions[:, mut_num], mut_res_idxs[:, mut_num]]
        
            #normalize wt scores by unbound likelihoods
            wt_scores = logits_bound[torch.arange(len(logits_bound)), mut_positions[:, mut_num], wt_res_idxs[:, mut_num]] - logits_unbound[torch.arange(len(logits_unbound)), mut_positions[:, mut_num], wt_res_idxs[:, mut_num]]

            #normalize mut scores by wt scores
            scores[:, mut_num] = mut_scores - wt_scores

        #ignore dummy mutation position
        scores = torch.where(padded_mutations_mask, 0, scores)
        scores = torch.sum(scores, dim = -1)

        #score for wt examples should be 0
        scores[wt_example_mask] = 0

    else:
        raise ValueError(f'Incorrect scoring method given: {method}, choose between: single, multiple')

    return scores
