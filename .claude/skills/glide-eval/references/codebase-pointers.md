# Codebase Pointers

기존 코드베이스에서 재활용한 코드와 Glide 파이프라인 구현체의 위치. 수정이나 확장 시 참조한다.

## 구조 I/O

| 기능 | 위치 | 핵심 함수 |
|------|------|----------|
| CIF 읽기 | allatom_design/utils/sample_io_utils.py | `load_example_with_parse()` |
| CIF 쓰기 | allatom_design/utils/sample_io_utils.py | `save_cif_file()` |
| PDB 쓰기 | atomworks/src/atomworks/io/utils/io_utils.py | `to_pdb_string()`, `to_pdb_buffer()` |
| 범용 읽기 | atomworks/src/atomworks/io/utils/io_utils.py | `load_any()`, `read_any()` |
| RDKit 변환 | atomworks/src/atomworks/io/tools/rdkit.py | `atom_array_to_rdkit()`, `atom_array_from_rdkit()` |
| SDF 읽기 | atomworks/src/atomworks/io/tools/rdkit.py | `sdf_to_rdkit()` |

SDF 쓰기는 atomworks에 없지만, `atom_array_to_rdkit()` 후 `rdkit.Chem.MolToMolFile()`로 가능.

## 구조 처리

| 기능 | 위치 | 방법 |
|------|------|------|
| 단백질 추출 | atomworks enums | `atom_array.chain_type == ChainType.POLYPEPTIDE_L` |
| 리간드 추출 | eval_utils/seq_des_utils.py | `extract_ligand_from_structure()` 또는 `np.isin(atom_array.pn_unit_iid, ligand_ids)` |
| Pocket annotation | data/transform/custom_transforms.py | `annotate_ligand_pockets()` |
| Atom selection | atomworks/io/utils/selection.py | `AtomSelection`, `get_mask_from_selection_string()` |
| ChainType enum | atomworks/src/atomworks/enums.py | `POLYPEPTIDE_L=6, NON_POLYMER=8` 등 |

## RMSD 및 Alignment

| 기능 | 위치 | 핵심 함수 |
|------|------|----------|
| Docking metrics (RMSD 포함) | eval_utils/eval_metrics.py | `compute_docking_metrics_atomarray()` |
| BS superposition + ligand RMSD | eval_utils/eval_metrics.py | `calculate_ligand_rmsd_with_binding_site_superposition()` |
| Structure alignment | atomworks/ml/utils/geometry.py | `align_atom_arrays()` |
| Symmetry-corrected RMSD | rdkit | `rdMolAlign.CalcRMS()` |

## Eval 패턴 (기존 eval 스크립트 참조)

| 패턴 | 위치 | 참조 |
|------|------|------|
| Hydra config + eval skeleton | eval/run_sc_eval.py, eval/run_tc_eval_af3.py | `@hydra.main` 패턴 |
| Array job parallelization | eval_utils/eval_setup_utils.py | `get_pdb_files()` - array_id/num_arrays |
| WandB logging | eval_utils/eval_setup_utils.py | `wandb_setup()` |
| AF3 confidence metrics | eval_utils/eval_metrics.py | `extract_af3_confidence_metrics()` |

## 작업 디렉토리

```
debug/260325_glide_debug/
├── example_data/           # AF3 predicted structures (CIF + JSON)
│   ├── 0H7_len_150_0_sample0.json
│   └── 0H7_len_150_0_sample0_seed-42_sample-0_model_pocket_aligned.cif
```

## Reference data

- De novo designs: denovo_val_cifs/ — CIF + JSON 쌍 (0H7_len_150_*_model_0.cif)
- Native crystals: native_val_cifs/cifs/ — PDB code.cif 형식 (1m5e.cif, 2ayr.cif 등)
- Native metadata: native_val_cifs/metadata_for_training_nativeval_sm_*.parquet

## Glide 파이프라인 구현체

| 모듈 | 위치 | 핵심 함수/클래스 |
|------|------|----------------|
| 전처리 | eval/glide/preprocessing.py | `preprocess_structure()`, `get_protein_pn_unit_iids()`, `get_ligand_pn_unit_iids()`, `write_ligand_sdf()`, `compute_ligand_centroid()`, `compute_dynamic_outerbox()` |
| Schrodinger 래퍼 | eval/glide/schrodinger_runner.py | `find_schrodinger()`, `run_prepwizard()`, `write_gridgen_input()`, `run_grid_generation()`, `run_ligprep()`, `write_docking_input()`, `run_glide()` |
| 결과 파싱 | eval/glide/result_parser.py | `parse_glide_csv()`, `parse_glide_sdf()`, `extract_best_scores()`, `get_pose_coordinates()` |
| 단일 샘플 오케스트레이션 | eval/glide/pipeline.py | `evaluate_single_sample()`, `run_glide_evaluation()`, `_run_inplace_scoring()`, `_run_redocking()`, `_compute_rmsd_vs_reference()` |
| 단일 샘플 진입점 | eval/glide/run_glide_eval.py | `main()` (Hydra), `build_reference_map()` |
| 샘플 선택 | eval/glide/sample_selection.py | `load_af3_metrics()`, `select_best_diffusion()`, `find_af3_prediction_path()` |
| 배치 오케스트레이션 | eval/glide/run_glide_eval_batch.py | `run_prep()`, `run_docking()`, `process_single_sample()`, `evaluate_batch()`, `main()` |

### 재활용한 기존 코드

| 기능 | 재활용 위치 | Glide 파이프라인에서의 사용 |
|------|-----------|-------------------------|
| `load_example_with_parse()` | sample_io_utils.py | preprocessing.py에서 CIF 로딩 |
| `to_pdb_string()` | atomworks io_utils.py | preprocessing.py에서 protein PDB 쓰기 |
| `atom_array_to_rdkit()` | atomworks rdkit.py | preprocessing.py에서 ligand SDF 변환 |
| `calculate_ligand_rmsd_with_binding_site_superposition()` | eval_metrics.py | pipeline.py에서 RMSD Mode 3 구현 |
| `get_pdb_files()` | eval_setup_utils.py | run_glide_eval.py에서 샘플 경로 로딩 |
| `wandb_setup()` | eval_setup_utils.py | run_glide_eval.py에서 로깅 설정 |
| `ChainType`, `ChainTypeInfo` | atomworks enums.py | preprocessing.py에서 체인 유형 분류 |
