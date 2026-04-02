# Implementation Decisions — Glide Evaluation Pipeline

이 문서는 Glide evaluation 파이프라인의 모든 설계 결정과 그 근거를 기록한다.
질문에 답할 때 반드시 이 문서를 참조하여 사실에 기반한 답변을 제공한다.

---

## 1. 아키텍처: 왜 5개 모듈로 분리했는가

```
preprocessing.py      → 구조 I/O (atomworks 영역)
schrodinger_runner.py → 외부 도구 호출 (subprocess 영역)
result_parser.py      → 출력 파싱 (pandas/RDKit 영역)
pipeline.py           → 오케스트레이션 (비즈니스 로직)
run_glide_eval.py     → Hydra 진입점 (config/CLI 영역)
```

**근거:**
- **관심사 분리**: Schrodinger 의존성은 schrodinger_runner.py에만 격리. preprocessing.py는 atomworks/RDKit만, result_parser.py는 pandas/RDKit만 사용한다. 이렇게 하면 Schrodinger 없이도 전처리와 파싱을 독립적으로 테스트할 수 있다.
- **테스트 용이성**: schrodinger_runner의 함수만 mock하면 나머지 모듈은 실제 로직으로 테스트 가능하다. 실제로 47개 테스트 중 Schrodinger가 필요한 것은 0개이다.
- **기존 패턴 준수**: run_sc_eval_af3.py → eval_utils → folding_utils 패턴과 동일한 계층 구조.

**대안 검토:**
- 단일 파일 구현 → 1000줄 이상이 되어 유지보수 어려움. 기존 프로젝트의 eval 스크립트도 utils로 분리되어 있음.
- Schrodinger Python API 직접 임포트 → lullaby_local 환경에 Schrodinger 패키지가 없어 불가. 환경 분리 원칙 위반.

---

## 2. Schrodinger 도구 호출: 왜 subprocess인가

**결정:** 모든 Schrodinger 도구를 `subprocess.run()`으로 호출한다.

**근거:**
- lullaby_local conda 환경에 Schrodinger Python 패키지가 설치되어 있지 않다 (environment-config.md의 "Python 환경 원칙" 참조).
- Schrodinger Python API가 필요한 작업은 `$SCHRODINGER/run`으로 별도 프로세스에서 실행해야 한다.
- PrepWizard, LigPrep, Glide는 모두 독립 실행 가능한 CLI 도구이다.
- subprocess 호출 시 `capture_output=True, text=True`로 stdout/stderr를 캡처하고, 실패 시 CalledProcessError를 raise한다.

**`-WAIT` 플래그:**
- Schrodinger 도구는 기본적으로 job을 submit하고 바로 리턴할 수 있다.
- `-WAIT`를 붙여 동기 실행을 보장한다.
- SLURM 환경에서는 각 array job이 소수의 샘플을 처리하므로 동기 실행이 적절하다.

**`-OVERWRITE` 플래그:**
- 같은 디렉토리에서 재실행 시 기존 출력 파일과 충돌을 방지한다.

---

## 3. 전처리: CIF → PDB + SDF

### 단백질: CIF → PDB
- `load_example_with_parse()` → `to_pdb_string()` → 파일 쓰기.
- PrepWizard가 PDB와 MAE를 모두 지원하므로 PDB를 선택. PDB는 범용적이고 디버깅이 쉽다.
- `to_pdb_string()`은 biotite의 PDB 라이터를 사용하며, 표준 단백질 구조에 대해 안정적이다.

### 리간드: CIF → SDF
- `atom_array_to_rdkit(sanitize=True)` → `Chem.SDWriter()` → SDF 파일.
- `Chem.MolToMolFile()` 대신 `SDWriter`를 사용한 이유: Glide의 LIGANDFILE 키워드가 SDF 형식을 기대. SDWriter는 $$$$​ delimiter를 올바르게 추가한다.
- sanitize 실패 시 `sanitize=False`로 재시도한다. 일부 CCD 코드의 분자가 RDKit sanitization을 통과하지 못하는 경우가 있기 때문이다.

### 리간드 centroid
- 리간드 heavy atom 좌표의 산술 평균으로 계산한다.
- 이 centroid가 Glide grid의 GRID_CENTER로 사용된다.
- 수소 원자를 제외하는 이유: 수소 위치가 부정확할 수 있고, heavy atom만으로 binding site 중심을 더 안정적으로 잡을 수 있다.

### 체인 자동 감지
- `receptor_pn_unit_iids`와 `ligand_pn_unit_iids`를 명시하지 않으면 자동 감지한다.
- 단백질: `chain_type == ChainType.POLYPEPTIDE_L`
- 리간드: `chain_type in ChainTypeInfo.NON_POLYMERS`
- 기존 코드베이스(run_sc_eval_af3.py의 `extract_pdb_chain_info()`)와 동일한 로직을 사용한다.

---

## 4. Glide 입력 파일 형식

### Grid generation `.in` 파일
```
FORCEFIELD   OPLS_2005
GRID_CENTER   x, y, z
INNERBOX   10, 10, 10
OUTERBOX   30.0, 30.0, 30.0
RECEP_FILE   receptor.mae
```
- INNERBOX [10,10,10]: 리간드 centroid 배치 가능 영역. 대부분의 소분자 리간드에 적합한 크기.
- OUTERBOX [30,30,30]: 그리드 전체 크기. receptor 주변을 충분히 커버.
- FORCEFIELD OPLS_2005: Glide의 기본 포스필드이며 가장 광범위하게 검증됨.

### Docking `.in` 파일
```
FORCEFIELD   OPLS_2005
GRIDFILE     grid.zip
LIGANDFILE   ligand.sdf
PRECISION    SP
DOCKING_METHOD   [inplace|confgen]
NREPORT      5
POSE_OUTTYPE   ligandlib_sd
COMPRESS_POSES   FALSE
WRITE_CSV    TRUE
```

**POSE_OUTTYPE = ligandlib_sd 선택 근거:**
- `poseviewer` (기본) → MAE 포맷 → Schrodinger 없이 읽기 어려움.
- `ligandlib_sd` → SDF 포맷 → RDKit으로 lullaby_local에서 직접 읽기 가능.
- Re-docking 후 pose RMSD를 계산하려면 좌표 접근이 필요한데, SDF가 가장 접근성이 좋다.

**COMPRESS_POSES = FALSE:**
- 디버깅 편의를 위해 비압축. .sdfgz도 파싱 코드에서 지원하므로 production에서는 TRUE로 변경 가능.

**WRITE_CSV = TRUE:**
- Glide 스코어를 CSV로 직접 출력. MAE 파일을 파싱하는 별도 Schrodinger 스크립트 없이 pandas로 바로 읽을 수 있다.

---

## 5. 세 가지 평가 모드의 설계 의도

### Mode 1: In-place Scoring (DOCKING_METHOD = mininplace)
- **목적**: AF3가 예측한 리간드 포즈의 에너지를 평가.
- **동작**: 리간드에 제한된 local minimization을 수행한 후 scoring한다 (mininplace).
- **의미**: Docking Score가 높으면(덜 음수이면) AF3 pose가 에너지적으로 불안정하다는 뜻. 10000 (penalty score)은 심각한 steric clash를 의미.
- PoseX 2025-3 프로토콜에 따라 `mininplace`를 사용한다. `inplace` (rigid scoring)는 AF3 pose quality 측정에 적합하지만, 목적이 "에너지 평가"이므로 local minimization을 허용한다.
- **`inplace` vs `mininplace`**: `inplace`는 좌표 이동 없이 scoring만 수행. `mininplace`는 ~0.5-2Å 범위의 local minimization 후 scoring. 에너지 평가가 목적이면 mininplace가 적절하다.

### Mode 2: Re-docking (DOCKING_METHOD = confgen)
- **목적**: AF3가 예측한 receptor에 대해 "최선의" 리간드 pose가 무엇인지 확인.
- **동작**: 리간드의 conformer를 새로 생성하고 유연 docking을 수행한다.
- **의미**: in-place score와 re-docking best score의 차이가 크면, AF3 pose가 최적이 아님을 의미. 반대로 차이가 작으면 AF3 pose가 이미 좋은 pose.
- PRECISION = SP를 기본으로 사용: XP는 더 정확하지만 10배 이상 느림. SP로 screening 후 필요 시 XP로 전환 가능.

### Mode 3: RMSD Comparison
- **목적**: AF3 예측 pose를 reference structure (crystal 또는 de novo design)과 비교.
- **구현**: 기존 eval_metrics.py의 `calculate_ligand_rmsd_with_binding_site_superposition()`을 재활용.
- **재활용 근거**: 이 함수는 이미 binding site superposition + symmetry-corrected RMSD를 구현하고 있으며, 프로젝트 내에서 검증되었다.
- Schrodinger가 전혀 필요 없는 순수 Python/atomworks/RDKit 연산이다.

### 세 모드가 독립적인 이유:
- config에서 각 모드를 개별적으로 on/off 가능.
- in-place scoring만 빠르게 돌리고 싶을 때 re-docking을 끌 수 있다.
- reference structure가 없으면 RMSD는 자동으로 skip된다.

---

## 6. 파이프라인 오케스트레이션 (pipeline.py)

### evaluate_single_sample() 흐름
```
1. preprocess_structure()  → protein.pdb + ligand.sdf + centroid
2. run_prepwizard()        → protein_prepared.mae
3. write_gridgen_input()   → gridgen.in
   run_grid_generation()   → gridgen.zip
4. run_ligprep() (optional)→ ligand_prepared.sdf
5. In-place scoring        → dock_inplace.csv → inplace_* 메트릭
6. Re-docking              → dock_redock.csv + dock_redock_lib.sdf → redock_* 메트릭
7. RMSD comparison         → rmsd_* 메트릭
```

**각 step이 실패해도 다음 step에 영향을 주지 않는 설계:**
- Step 5, 6, 7은 독립적인 try/except로 감싼다.
- Step 5가 실패해도 Step 6은 실행된다.
- 실패 시 `metrics["inplace_error"]`에 에러 메시지를 기록하고 계속 진행.

**LigPrep가 optional인 이유:**
- AF3 입력 리간드는 CCD 코드에서 생성되므로 이미 합리적인 3D 구조를 가진다.
- LigPrep는 ionization 상태, tautomer 등을 최적화하지만 in-place scoring에서는 원래 구조를 유지하는 것이 목적.
- Re-docking에서는 LigPrep를 켜면 더 나은 결과를 얻을 수 있다.

### run_glide_evaluation() — 배치 처리
- 기존 `get_pdb_files()`의 `array_id/num_arrays` 패턴을 그대로 사용하여 SLURM array job 병렬화 가능.
- 실패한 샘플은 `glide_failed_samples.txt`에 기록하여 재실행 시 활용 가능.

---

## 7. 결과 파싱 (result_parser.py)

### CSV 컬럼 매핑
- Schrodinger 내부 property 이름(`r_i_glide_gscore`)을 사람이 읽기 쉬운 이름(`glide_score`)으로 변환.
- 매핑에 없는 컬럼은 원본 이름 그대로 유지한다 (확장성).

### SDF 파싱
- `Chem.ForwardSDMolSupplier`에 **binary mode** 파일 핸들을 전달해야 한다.
- 초기 구현에서 text mode(`"r"`)를 사용했다가 테스트에서 `ValueError: Need a binary mode file object`가 발생하여 `"rb"`로 수정.
- `.sdfgz` 파일도 `gzip.open(path, "rb")`로 동일하게 처리.

### GlideScore vs Docking Score
- `r_i_glide_gscore` (GlideScore): Glide의 scoring function 값. 수소결합, 소수성 등의 상호작용 항의 합.
- `r_i_docking_score` (Docking Score): GlideScore + Epik state penalty. 최종 랭킹에 사용.
- `r_i_glide_emodel` (Emodel): 포즈 선별에 사용되는 에너지 모델 스코어. 랭킹보다는 포즈 selection에 적합.

---

## 8. Hydra Config 설계

### 2-layer config
- `configs/eval/glide/run_glide_eval.yaml`: 기본값 (환경 독립적).
- `configs_local/eval/glide/run_glide_eval.yaml`: 로컬 환경 override (Schrodinger 경로, 데이터 경로 등).

**근거:** 기존 프로젝트의 configs/ + configs_local/ 패턴을 따른다. configs/는 git에 커밋되고, configs_local/은 환경별로 다를 수 있다.

### config_path 선택
- `run_glide_eval.py`의 `@hydra.main`에서 `config_path="../../configs_local/eval/glide"`를 사용.
- 기존 `run_sc_eval_af3.py`가 `config_path="../configs_local/eval"`을 사용하는 패턴과 동일.
- 로컬 config가 기본 config를 `defaults`로 상속한다.

### PrepWizard 기본 옵션
```yaml
prepwizard:
  propka_pH: 7.4
  f: S-OPLS
```
- PoseX 2025-3 프로토콜에 따라 restrained minimization을 켠다 (noimpref 제거).
- S-OPLS forcefield 사용 (PrepWizard/LigPrep 단계). Grid generation과 docking은 OPLS4를 사용한다.
- propka_pH 7.4: 생리학적 pH에서의 protonation state 예측.

---

## 9. 테스트 전략

### 원칙: Schrodinger 없이 모든 테스트 통과
- Schrodinger 호출은 전부 mock한다.
- 전처리, 파싱, 오케스트레이션 로직은 실제 코드로 테스트한다.

### 테스트 구성 (47개)
| 파일 | 개수 | 테스트 대상 | Schrodinger 필요 |
|------|------|------------|-----------------|
| test_preprocessing.py | 10 | 체인 감지, centroid, SDF 쓰기, CIF 통합 | No (atomworks만) |
| test_schrodinger_runner.py | 13 | .in 파일 생성, find_schrodinger, mock 실행 | No (mock) |
| test_result_parser.py | 12 | CSV 파싱, SDF 파싱, score 추출, gzip 처리 | No |
| test_pipeline.py | 5 | 파이프라인 오케스트레이션, 에러 핸들링, 배치 | No (mock) |

### @requires_example_data / @requires_schrodinger
- 예제 CIF가 없으면 통합 테스트를 skip한다 (CI 환경 호환).
- Schrodinger가 없으면 end-to-end 테스트를 skip한다.
- `conftest.py`에서 경로 존재 여부로 판단.

### fixture 설계
- `sample_glide_csv`: 인라인 CSV 문자열로 Glide 출력 모사.
- `sample_glide_sdf`: RDKit으로 benzene을 생성하여 SDF 출력 모사.
- `mock_schrodinger_path`: tmp_path에 가짜 실행 파일 생성.

---

## 10. 기존 코드베이스 재활용 목록

| 기능 | 재활용한 코드 | 위치 |
|------|-------------|------|
| CIF 파싱 | `load_example_with_parse()` | sample_io_utils.py |
| PDB 쓰기 | `to_pdb_string()` | atomworks/io/utils/io_utils.py |
| RDKit 변환 | `atom_array_to_rdkit()` | atomworks/io/tools/rdkit.py |
| RMSD 계산 | `calculate_ligand_rmsd_with_binding_site_superposition()` | eval_metrics.py |
| ChainType enum | `ChainType.POLYPEPTIDE_L`, `ChainTypeInfo.NON_POLYMERS` | atomworks/enums.py |
| 샘플 경로 로딩 | `get_pdb_files()` | eval_setup_utils.py |
| WandB 세팅 | `wandb_setup()` | eval_setup_utils.py |
| Hydra 패턴 | `@hydra.main`, config override | run_sc_eval_af3.py |

**새로 작성한 코드**: Schrodinger CLI 래퍼, Glide .in 파일 생성, CSV/SDF 결과 파싱, 파이프라인 오케스트레이션.

---

## 11. 알려진 제약사항과 향후 개선점

### 현재 제약
1. **단일 리간드 가정**: 여러 리간드가 있는 구조에서는 auto-detect가 모든 NON_POLYMER를 잡는다. 특정 리간드만 평가하려면 `ligand_pn_unit_iids`를 명시해야 한다.
2. **XP precision 미지원**: config에서 precision을 변경할 수 있지만, XP는 Glide 버전과 라이선스에 따라 제한될 수 있다.
3. **PDB 형식 한계**: 비표준 잔기, 수정된 아미노산 등은 PDB 형식에서 정보가 손실될 수 있다. 중요한 경우 CIF → MAE 직접 변환이 필요할 수 있다.

### 향후 개선
1. **SLURM sbatch 스크립트 생성**: 현재는 run_glide_eval.py를 직접 실행. SLURM 스크립트 자동 생성 추가 가능.
2. **Covalent docking 지원**: 공유결합 리간드에 대한 별도 워크플로우.
3. **결과 시각화**: Glide score 분포, RMSD scatter plot 등.
4. **Re-docking pose vs reference RMSD**: redocked SDF pose를 원본 sample의 ligand과 비교하는 pocket-aligned ligand RMSD.

---

## 12. 배치 평가 파이프라인 (run_glide_eval_batch.py)

### 아키텍처

```
AF3 eval step dir (af3_eval_dir)
    ├── all_docking_metrics_per_designed_sample.csv
    ├── all_sc_metrics_per_designed_sample.csv
    ├── af3_ss_preds/
    │    └── {designed_id}/seed-42_sample-{X}/*_pocket_aligned.cif
    └── samples/

    ↓ [Phase 1] load_af3_metrics() + select_best_diffusion()

selected_df: best diffusion per designed_sample (cutoff-filtered)

    ↓ [Phase 2] evaluate_batch() → ProcessPoolExecutor

Per sample: run_prep() → run_docking(mininplace) → run_docking(redocking)

    ↓ Output

all_glide_metrics_per_designed_sample_{suffix}.csv
```

### 모듈화 설계

기존 모듈의 함수를 재조합하여 배치 파이프라인을 구현했다:
- `preprocessing.py`의 `preprocess_structure()`, `compute_dynamic_outerbox()`
- `schrodinger_runner.py`의 모든 함수
- `result_parser.py`의 `parse_glide_csv()`, `extract_best_scores()`

새로 추가한 모듈:
- `sample_selection.py`: AF3 metrics CSV 파싱, cutoff filtering, best diffusion 선택
- `run_glide_eval_batch.py`: `run_prep()`, `run_docking()`, `process_single_sample()`, `evaluate_batch()`

`process_single_sample()`은 stateless 함수로 설계하여 `ProcessPoolExecutor`로 병렬화 가능.

### Sample selection 로직
1. `load_af3_metrics()`: docking CSV + SC CSV를 읽어 dict-string 컬럼을 per-diffusion 행으로 flatten.
2. `select_best_diffusion()`: `ligand_rmsd <= cutoff` AND `ligand_plddt >= cutoff`를 만족하는 diffusion 중, 각 designed_sample_id에서 `ligand_plddt`가 가장 높은 것을 선택.

### Output CSV suffix
cutoff 값을 파일명에 반영: `_lplddt{plddt}_lrmsd{rmsd}` (예: `_lplddt70_lrmsd2`).
어떤 cutoff로 생성된 결과인지 파일명만으로 구분 가능하다.

### Debug mode
`debug: true` + `num_debug_samples: N`으로 선택된 샘플 중 상위 N개만 처리. 로컬 테스트에서 빠른 검증 용도.

### AF3 prediction 경로 해석
`find_af3_prediction_path()`는 세 가지 전략으로 CIF를 찾는다:
1. exact match: `{af3_preds_dir}/{designed_id}/seed-42_sample-{X}/*_pocket_aligned.cif`
2. `len_150` → `len150` 변환 후 재시도 (폴더 이름 불일치 대응)
3. glob fallback: CCD 코드로 glob 검색
