import pandas as pd
import json

DISTANCE_CUTOFF = 5.0

METADATA_PATHS = [
    "/home/possu/jinho/datasets/atomworks_pdb_full_v3/metadata_seq_clustered_04_filtered_250205.parquet",
    "/home/possu/jinho/datasets/atomworks_pdb_full_v3/debug_metadata_seq_clustered_04_filtered_250205.parquet",
]

def get_context_group_and_protein_count(row, pn_unit_is_protein_map, distance_cutoff):
    """
    자기 자신(small molecule chain) + 5Å 이내 contacting하고 있는 모든 chain의
    pn_unit_iid 리스트와, 그 중 protein chain의 개수를 반환.
    """
    contacts = row['q_pn_unit_contacting_pn_unit_iids']
    pdb_id = row['pdb_id']
    assembly_id = row['assembly_id']
    
    # 자기 자신을 먼저 포함
    context_group = [row['q_pn_unit_iid']]
    num_protein = 0
    
    for contact in contacts:
        pn_unit_iid = contact.get('pn_unit_iid')
        min_distance = contact.get('min_distance')
        
        if min_distance is not None and min_distance <= distance_cutoff:
            context_group.append(pn_unit_iid)
            
            key = f"{pdb_id}_{assembly_id}_{pn_unit_iid}"
            if pn_unit_is_protein_map.get(key, False):
                num_protein += 1
    
    return (context_group, num_protein)

def process_metadata(metadata_path):
    print(f"\n{'='*60}")
    print(f"Processing: {metadata_path}")
    print(f"{'='*60}")
    
    metadata_df = pd.read_parquet(metadata_path)
    metadata_df['q_pn_unit_contacting_pn_unit_iids'] = metadata_df['q_pn_unit_contacting_pn_unit_iids'].apply(json.loads)
    
    # (pdb_id, assembly_id, pn_unit_iid) -> is_protein 매핑 생성
    metadata_df['_key'] = (
        metadata_df['pdb_id'] + '_' + 
        metadata_df['assembly_id'].astype(str) + '_' + 
        metadata_df['q_pn_unit_iid']
    )
    pn_unit_is_protein_map = metadata_df.set_index('_key')['q_pn_unit_is_protein'].to_dict()
    
    bm_sm_mask = metadata_df['q_pn_unit_is_biologically_meaningful_small_molecule']
    
    # biologically meaningful small molecule에 대해서만 적용
    print(f"Processing {bm_sm_mask.sum()} biologically meaningful small molecule entries...")
    
    metadata_df['q_pn_unit_context_group_iids'] = None
    metadata_df['num_contacting_protein_chains'] = 0
    
    results = metadata_df.loc[bm_sm_mask].apply(
        get_context_group_and_protein_count, 
        axis=1, 
        pn_unit_is_protein_map=pn_unit_is_protein_map,
        distance_cutoff=DISTANCE_CUTOFF
    )
    metadata_df.loc[bm_sm_mask, 'q_pn_unit_context_group_iids'] = results.apply(lambda x: x[0])
    metadata_df.loc[bm_sm_mask, 'num_contacting_protein_chains'] = results.apply(lambda x: x[1])
    
    # 임시 컬럼 제거
    metadata_df.drop(columns=['_key'], inplace=True)
    
    # 저장
    output_path = metadata_path.replace('_250205.parquet', '_grouped_250205.parquet')
    metadata_df.to_parquet(output_path)
    print(f"Saved to {output_path}")
    
    # 결과 확인
    bm_df = metadata_df[bm_sm_mask]
    bm_non_cov_df = bm_df[~bm_df['q_pn_unit_is_maybe_covalently_linked_to_protein']]
    
    print(f"\n=== [All biologically meaningful small molecules] ===")
    print(f"총 수: {len(bm_df)}")
    print(f"5Å 이내 neighbor chain이 있는 수: {(bm_df['q_pn_unit_context_group_iids'].apply(len) > 1).sum()}")
    print(f"\nnum_contacting_protein_chains 분포:")
    print(bm_df['num_contacting_protein_chains'].value_counts().sort_index())
    
    print(f"\n=== [Non-covalently linked biologically meaningful small molecules] ===")
    print(f"총 수: {len(bm_non_cov_df)}")
    print(f"5Å 이내 neighbor chain이 있는 수: {(bm_non_cov_df['q_pn_unit_context_group_iids'].apply(len) > 1).sum()}")
    print(f"\nnum_contacting_protein_chains 분포:")
    print(bm_non_cov_df['num_contacting_protein_chains'].value_counts().sort_index())

if __name__ == '__main__':
    for path in METADATA_PATHS:
        process_metadata(path)
