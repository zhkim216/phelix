# Codebase Pointers

PoseBusters evaluation 코드 작성 시 재활용할 기존 코드와 참조할 패턴의 위치.

## 구조 I/O (기존 코드 재활용)

| 기능 | 위치 | 핵심 함수 |
|------|------|----------|
| CIF 읽기 | allatom_design/utils/sample_io_utils.py | `load_example_with_parse()` |
| PDB 쓰기 | atomworks/src/atomworks/io/utils/io_utils.py | `to_pdb_string()`, `to_pdb_buffer()` |
| RDKit 변환 | atomworks/src/atomworks/io/tools/rdkit.py | `atom_array_to_rdkit()` |
| SDF 읽기 | atomworks/src/atomworks/io/tools/rdkit.py | `sdf_to_rdkit()` |

SDF 쓰기: `atom_array_to_rdkit()` 후 `rdkit.Chem.MolToMolFile()`

## 구조 처리

| 기능 | 위치 | 방법 |
|------|------|------|
| 단백질 추출 | atomworks enums | `atom_array.chain_type == ChainType.POLYPEPTIDE_L` |
| 리간드 추출 | atomworks enums | `atom_array.chain_type == ChainType.NON_POLYMER` |
| 리간드 추출 (pn_unit_iid) | eval_utils/seq_des_utils.py | `extract_ligand_from_structure()` |
| Pocket annotation | data/transform/custom_transforms.py | `annotate_ligand_pockets()` |
| ChainType enum | atomworks/src/atomworks/enums.py | `POLYPEPTIDE_L=6, NON_POLYMER=8` |

## CIF → PDB/SDF 변환 (Glide preprocessing 참조)

**핵심 파일**: `allatom_design/eval/glide/preprocessing.py`

이 파일이 CIF에서 protein PDB와 ligand SDF를 분리하는 전체 로직을 포함한다:
- `preprocess_structure()`: CIF 로드 → protein/ligand 분리 → PDB/SDF 저장
- `get_protein_pn_unit_iids()`: protein chain 식별
- `get_ligand_pn_unit_iids()`: ligand chain 식별
- `write_ligand_sdf()`: ligand AtomArray → RDKit Mol → SDF 파일
- `compute_ligand_centroid()`: ligand 중심 좌표 계산

이 함수들을 직접 import하거나, 패턴을 참고하여 eval_posebusters.py에 맞게 조정한다.

## RMSD 및 Alignment

| 기능 | 위치 | 핵심 함수 |
|------|------|----------|
| Docking metrics (RMSD 포함) | eval_utils/eval_metrics.py | `compute_docking_metrics_atomarray()` |
| BS superposition + ligand RMSD | eval_utils/eval_metrics.py | `calculate_ligand_rmsd_with_binding_site_superposition()` |
| Symmetry-corrected RMSD | rdkit | `rdMolAlign.CalcRMS()` |

## Eval 패턴 (기존 eval 스크립트 참조)

| 패턴 | 위치 | 참조 |
|------|------|------|
| Hydra config + eval skeleton | eval/run_sc_eval.py, eval/run_tc_eval_af3.py | `@hydra.main` 패턴 |
| Array job parallelization | eval_utils/eval_setup_utils.py | `get_pdb_files()` - array_id/num_arrays |
| WandB logging | eval_utils/eval_setup_utils.py | `wandb_setup()` |
| Config override (local) | configs_local/ | 환경별 config override |

## Glide 파이프라인 (통합 대상)

| 모듈 | 위치 | 핵심 함수 |
|------|------|----------|
| 오케스트레이션 | eval/glide/pipeline.py | `evaluate_single_sample()`, `_run_inplace_scoring()`, `_run_redocking()` |
| 진입점 | eval/glide/run_glide_eval.py | `main()` (Hydra) |
| 결과 파싱 | eval/glide/result_parser.py | `parse_glide_csv()`, `extract_best_scores()` |

Glide pipeline의 mininplace/redocking 출력(SDF)을 PoseBusters에 전달하여 validity check.

## 외부 레퍼런스 코드

| 프로젝트 | 위치 | PB 사용 파일 |
|---------|------|------------|
| PoseX | /home/possu/jinho/PoseX/ | `scripts/calculate_benchmark_result.py` (bust_table 배치), `scripts/complex_structure_alignment.py` (check_rmsd) |
| plinder | /home/possu/jinho/allatom-design/plinder/ | `src/plinder/eval/docking/utils.py` (bust 단일, ModelScores 클래스) |

## 새로 작성할 파일 위치

```
allatom_design/eval/eval_utils/eval_posebusters.py    # 핵심 로직
allatom_design/configs/eval/posebusters/               # Hydra config (필요 시)
```
