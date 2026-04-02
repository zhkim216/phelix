---
name: glide-eval
description: "[DEPRECATED — ligand-eval로 통합됨] 이 스킬을 직접 트리거하지 말 것. Glide 관련 요청은 ligand-eval 스킬이 처리한다. 이 디렉토리의 references/만 참조용으로 유지."
---

# Glide Evaluation Pipeline (Archived)

> **이 스킬은 `ligand-eval`로 통합되었다.** Glide 관련 요청은 `ligand-eval` 스킬을 사용한다.
> 이 디렉토리의 `references/` 파일들은 에이전트와 통합 스킬이 참조하므로 유지한다.

아래는 아카이브된 원본 내용이다.

## 원본 설명

AF3 predicted protein-ligand structure의 docking quality를 Schrodinger Glide로 평가하는 파이프라인을 조율한다. 또한 파이프라인의 설계 의도와 구현 근거를 정확하게 설명한다.

## 목표

AF3가 예측한 protein-ligand complex가 docking 관점에서 얼마나 좋은지 평가한다:
1. AF3 predicted pose를 있는 그대로 scoring (in-place)
2. 같은 receptor에 ligand를 re-docking하여 best pose와 비교
3. Reference crystal structure / de novo design 대비 RMSD 비교

## 입력 스펙

| 항목 | 형식 | 예시 |
|------|------|------|
| AF3 predicted structure | mmCIF (.cif) | `*_pocket_aligned.cif` |
| AF3 input metadata | JSON | protein sequence, ligand CCD codes, model seeds |
| Reference structures (optional) | mmCIF (.cif) | denovo_val_cifs/, native_val_cifs/ |

구조 파일에는 protein chain(s)과 ligand(s)가 함께 포함되어 있다.

## 출력 스펙

| 항목 | 형식 |
|------|------|
| Per-sample metrics | CSV/parquet (sample_id, glide_score, docking_score, ligand_rmsd_vs_ref, ...) |
| Glide 중간 산출물 | Schrodinger output files (각 sample별 서브디렉토리) |
| 요약 리포트 | 전체 통계, 실패 sample 목록 |

## 평가 모드

### Mode 1: In-place Scoring
AF3 predicted pose를 변경 없이 Glide로 scoring한다.
목적: AF3 pose quality의 Glide 관점 평가.

### Mode 2: Re-docking
AF3 predicted structure에서 receptor를 준비하고, ligand를 re-dock한다.
Glide SP 및/또는 XP precision.
목적: receptor quality 평가, best achievable score 확인.

### Mode 3: RMSD Comparison
AF3 predicted ligand pose vs reference structure의 ligand pose를 비교한다.
Binding site superposition 후 symmetry-corrected ligand RMSD.
Reference: native crystal structure 또는 de novo generated structure.

세 모드는 독립적으로 또는 함께 실행 가능하다.

## 환경 요구사항

### Config 관리
- 모든 환경별 경로는 config 파일로 관리한다.
- 같은 Python 코드가 로컬과 Sherlock에서 모두 동작해야 한다.

### Python 환경
- 메인 코드는 lullaby_local conda 환경에서 실행한다.
- 기존 환경의 패키지를 제거하거나 버전을 변경하지 않는다.
- Schrodinger Python API가 필요한 작업은 `$SCHRODINGER/run`으로 별도 프로세스에서 실행한다.

### 경로 (모두 config)
환경별 경로는 `references/environment-config.md`를 참조한다.

## 기존 코드 활용 원칙

이 프로젝트에는 이미 풍부한 구조 처리 유틸리티가 있다. 새로 작성하기 전에 기존 코드를 확인하라.
상세 위치는 `references/codebase-pointers.md`를 참조한다.

핵심:
- **구조 I/O**: atomworks의 parser/writer (CIF 읽기, PDB 쓰기)
- **리간드/단백질 분리**: atomworks의 chain_type, pn_unit_iid annotation
- **RDKit 변환**: atomworks의 atom_array_to_rdkit()
- **RMSD 계산**: eval_metrics.py의 compute_docking_metrics_atomarray()
- **Eval 패턴**: 기존 eval 스크립트의 Hydra config + array job 패턴

## 모듈화 & 재활용 원칙

코드를 작성할 때 **모듈화와 재활용**을 최우선으로 고려한다. 각 함수는 독립적으로 호출 가능해야 하며, 다른 파이프라인에서도 조합하여 사용할 수 있어야 한다.

### 핵심 규칙
1. **단일 책임**: 함수 하나는 하나의 작업만 수행한다. prep, docking, parsing, metric 계산을 하나의 함수에 섞지 않는다.
2. **config 독립적 인터페이스**: 함수 인자로 plain dict/기본 타입을 받는다. OmegaConf/Hydra 의존성은 진입점(main)에서만 처리하고, 내부 함수에 전파하지 않는다. 이렇게 하면 Hydra 없이도 Python에서 직접 호출 가능하다.
3. **조합 가능한 단위**: `run_prep()`, `run_docking()`, `process_single_sample()` 등 각 단위는 입력-출력이 명확하여 순서를 바꾸거나 일부만 사용할 수 있다. 예: prep만 미리 실행하고, docking은 나중에 별도로 실행 가능.
4. **기존 모듈 재활용 우선**: 새 기능 추가 시 기존 모듈의 함수를 조합하여 구현한다. `preprocessing.py`, `schrodinger_runner.py`, `result_parser.py`의 함수들은 `pipeline.py`와 `run_glide_eval_batch.py` 양쪽에서 동일하게 재활용된다.
5. **병렬화 친화적 설계**: per-sample 처리 함수(`process_single_sample()`)는 stateless하게 만들어 ProcessPoolExecutor로 바로 병렬화할 수 있어야 한다. 공유 상태나 전역 변수에 의존하지 않는다.

## 구현 완료 상태

파이프라인은 이미 구현되어 있다. 아래 파일들을 참조한다:

### 소스 코드
| 파일 | 역할 |
|------|------|
| `allatom_design/eval/glide/preprocessing.py` | CIF → 단백질 PDB + 리간드 SDF 분리, dynamic outerbox |
| `allatom_design/eval/glide/schrodinger_runner.py` | PrepWizard, Grid Gen, LigPrep, Glide subprocess 래퍼 |
| `allatom_design/eval/glide/result_parser.py` | Glide CSV/SDF 출력 파싱 |
| `allatom_design/eval/glide/pipeline.py` | 단일 샘플 평가 오케스트레이션 |
| `allatom_design/eval/glide/run_glide_eval.py` | 단일 샘플 Hydra 진입점 |
| `allatom_design/eval/glide/sample_selection.py` | AF3 metrics 로딩, cutoff 필터링, best diffusion 선택 |
| `allatom_design/eval/glide/run_glide_eval_batch.py` | 배치 평가 Hydra 진입점 (multiprocessing 지원) |

### Config
| 파일 | 역할 |
|------|------|
| `configs/eval/glide/run_glide_eval.yaml` | 단일 샘플 기본 config |
| `configs/eval/glide/run_glide_eval_batch.yaml` | 배치 평가 기본 config (cutoff, debug, num_workers) |
| `configs_local/eval/glide/run_glide_eval.yaml` | 단일 샘플 로컬 override |
| `configs_local/eval/glide/run_glide_eval_batch.yaml` | 배치 평가 로컬 override |

### 테스트 (47개, 전부 통과)
| 파일 | 역할 |
|------|------|
| `tests/glide/conftest.py` | fixture, skip marker |
| `tests/glide/test_preprocessing.py` | 전처리 로직 |
| `tests/glide/test_schrodinger_runner.py` | .in 파일 생성, mock 실행 |
| `tests/glide/test_result_parser.py` | CSV/SDF 파싱 |
| `tests/glide/test_pipeline.py` | 오케스트레이션, 에러 핸들링 |

### 설계 근거
모든 구현 결정의 상세한 근거는 `references/implementation-decisions.md`에 기록되어 있다.

## 에이전트 구성

| 에이전트 | subagent_type | 역할 | 출력 |
|---------|--------------|------|------|
| glide-engineer | glide-engineer | 코드 구현, 디버깅, 구현 의도 설명 | Python 코드, configs, 설명 |
| glide-analyst | glide-analyst | 결과 분석, 메트릭 해석, 분석 방법 설명 | 메트릭 CSV, 요약, 해석 |

## 워크플로우

### 구현/수정 요청 시

#### Phase 1: 요구사항 확인
1. 사용자 입력 분석 — 어떤 평가 모드가 필요한지, 입력 데이터 위치, 출력 위치
2. 환경 확인 — 로컬/Sherlock, config 경로
3. Batch 규모 확인 — 몇 개의 sample, 병렬 처리 필요 여부

#### Phase 2: 파이프라인 수정
glide-engineer가 기존 코드를 수정하거나 확장한다.

#### Phase 3: 결과 분석
glide-analyst가 결과를 분석한다.

#### Phase 4: 디버깅 (필요 시)
파이프라인 실행 중 문제 발생 시 glide-engineer가 디버깅한다.

### 질문 응답 시

사용자가 파이프라인에 대해 질문하면 다음 절차를 따른다:

1. **질문 분류**: 코드 구현, 설계 의도, Schrodinger 도구, 메트릭 해석 중 어느 영역인지 파악.
2. **근거 확인**: `references/implementation-decisions.md`를 읽어 해당 결정의 근거를 확인.
3. **코드 확인**: 실제 코드를 읽어 implementation-decisions.md의 내용과 일치하는지 검증.
4. **정직한 답변**: 모르는 것은 모른다고 답한다. 추측과 사실을 구분한다.

#### 답변 원칙
- **코드를 근거로**: "이렇게 구현했을 것이다"가 아니라 실제 코드를 읽고 "이렇게 구현되어 있다"로 답한다.
- **대안도 함께**: 선택한 방식의 근거뿐 아니라, 검토했지만 선택하지 않은 대안과 그 이유도 설명한다.
- **제약사항 솔직히**: 현재 구현의 한계를 숨기지 않는다. `implementation-decisions.md` §11 참조.
- **"왜" 중심**: 단순히 "무엇을 했다"가 아니라 "왜 그렇게 했다"를 중심으로 설명한다.

## 데이터 흐름

```
AF3 CIF + JSON
    |
[전처리] atomworks로 구조 읽기, 단백질/리간드 분리
    |
[Schrodinger pipeline] protein prep -> grid gen -> ligand prep -> docking/scoring
    |
[결과 수집] Glide output 파싱
    |
[메트릭 계산] RMSD vs reference, score 정리
    |
CSV/parquet + 요약 리포트
```

## 에러 핸들링

| 상황 | 전략 |
|------|------|
| Schrodinger 도구 실패 | stderr 로깅, 해당 sample skip, 요약에 실패 목록 포함 |
| 구조 변환 실패 | 에러 로깅, 대안 시도 |
| Reference structure 없음 | RMSD 계산 skip, 나머지 메트릭만 보고 |
| Batch 중 일부 실패 | 성공한 sample로 결과 생성, 실패 목록 별도 저장 |

## 테스트 시나리오

### 정상 흐름
1. example_data/의 0H7 sample로 전체 파이프라인 로컬 실행
2. In-place scoring -> GlideScore 산출
3. Re-docking -> best pose score 산출
4. RMSD vs denovo_val_cifs의 해당 reference -> ligand RMSD 산출
5. 결과 CSV 생성

### 에러 흐름
1. Schrodinger 경로가 잘못 설정된 경우 -> config 에러 메시지
2. CIF 파일에 리간드가 없는 경우 -> sample skip + 로그
3. PrepWizard가 특정 구조에서 실패하는 경우 -> sample skip + stderr 로그
