---
name: ligand-eval
description: "AF3 predicted structure의 ligand pose quality를 Glide docking scoring과 PoseBusters chemical/physical validity check로 종합 평가하는 통합 파이프라인 오케스트레이터. (1) 'Glide evaluation', 'docking evaluation', 'docking score 계산', 'in-place scoring', 're-docking', 'mininplace' 등 Glide 관련 요청, (2) 'PoseBusters', 'pb_valid', 'ligand validity', 'chemical check', 'physical check', 'bond length check', 'steric clash', 'volume overlap' 등 PB 관련 요청, (3) 'ligand eval', 'pose quality', 'docking quality', 'ligand quality' 등 통합 분석 요청, (4) Schrodinger 도구 사용법/파라미터 의미에 대한 질문, (5) PB 메트릭 의미/해석 질문 시 반드시 트리거. Glide와 PoseBusters 중 하나만 언급해도 이 스킬을 사용할 것."
---

# Ligand Evaluation — Glide + PoseBusters 통합 평가

AF3 predicted protein-ligand complex의 ligand pose quality를 두 가지 관점에서 종합 평가한다:
1. **Glide docking** — 에너지 기반 scoring (in-place, re-docking), RMSD 비교
2. **PoseBusters** — 화학적/물리적 plausibility (bond lengths, steric clash, ring flatness 등)

두 평가는 독립적으로 또는 함께 실행 가능하다. 통합 결과로 "좋은 docking score + 화학적으로 valid한 pose"를 종합 판단한다.

## 실행 모드: 서브 에이전트

## 평가 축

| 축 | 도구 | 핵심 메트릭 |
|----|------|-----------|
| 에너지 | Glide SP/XP | GlideScore, DockingScore, Emodel |
| Pose 재현성 | Glide re-docking | in-place vs re-dock score 차이 |
| 기하 정합성 | RMSD | ligand RMSD vs reference |
| 화학 validity | PoseBusters | bond_lengths, bond_angles, sanitization |
| 물리 validity | PoseBusters | steric_clash, ring_flatness, internal_energy |
| 상호작용 validity | PoseBusters | min_distance_to_protein, volume_overlap |
| 종합 | Glide + PB | pb_valid AND DockingScore < threshold |

## 입력 스펙

| 항목 | 형식 | 예시 |
|------|------|------|
| AF3 predicted structure | mmCIF (.cif) | `*_pocket_aligned.cif` |
| AF3 input metadata | JSON | protein sequence, ligand CCD codes, model seeds |
| Reference structures (optional) | mmCIF (.cif) | denovo_val_cifs/, native_val_cifs/ |

## 출력 스펙

| 항목 | 형식 |
|------|------|
| Glide per-sample metrics | CSV (glide_score, docking_score, ligand_rmsd, ...) |
| PB per-sample metrics | CSV (pb_valid, bond_lengths, steric_clash, ...) |
| 통합 메트릭 | CSV (Glide + PB columns merged per sample) |
| 실패 샘플 목록 | TXT |

## 에이전트 구성

| 에이전트 | subagent_type | 역할 | 출력 |
|---------|--------------|------|------|
| glide-engineer | glide-engineer | Glide pipeline 코드 구현/수정/디버깅/설명 | Python 코드, configs |
| pb-engineer | pb-engineer | PoseBusters evaluation 코드 구현/디버깅 | Python 코드, configs |
| glide-analyst | glide-analyst | Glide 결과 분석, 메트릭 해석 | 메트릭 CSV, 요약, 해석 |
| pb-analyst | pb-analyst | PB 결과 분석, validity 해석 | 메트릭 CSV, 요약, 해석 |

모든 Agent 호출 시 `model: "opus"` 파라미터를 명시한다.

## 워크플로우

### 구현/수정 요청 시

#### Phase 1: 요구사항 확인
1. 어떤 평가가 필요한지 (Glide만 / PB만 / 둘 다)
2. 입력 데이터 위치, 출력 위치
3. 환경 (로컬/Sherlock), config 경로
4. Batch 규모, 병렬 처리 필요 여부

#### Phase 2: 구현
요청 유형에 따라 적절한 engineer를 호출한다:

| 요청 | 에이전트 | run_in_background |
|------|---------|-------------------|
| Glide pipeline 관련 | glide-engineer | false |
| PB evaluation 관련 | pb-engineer | false |
| 공통 전처리 (CIF→PDB/SDF) | glide-engineer (preprocessing.py 소유) | false |
| 둘 다 수정 필요 | glide-engineer → pb-engineer (순차) | false |

#### Phase 3: 결과 분석
요청 유형에 따라 적절한 analyst를 호출한다:

| 요청 | 에이전트 | run_in_background |
|------|---------|-------------------|
| Glide 결과만 | glide-analyst | false |
| PB 결과만 | pb-analyst | false |
| 통합 분석 | glide-analyst + pb-analyst (병렬) → 결과 종합 | true (둘 다) |

통합 분석 시 두 analyst를 병렬 호출한 뒤, 결과를 종합하여 교차 인사이트를 제공한다:
- "RMSD 낮은데 pb_valid=False" → 화학적 문제 진단
- "GlideScore 좋은데 volume_overlap 실패" → steric clash 분석
- "mininplace에서 pb_valid 개선" → Glide minimization 효과 평가

#### Phase 4: 디버깅 (필요 시)
- Glide pipeline 문제 → glide-engineer
- PB 관련 문제 → pb-engineer
- 공통 전처리 문제 → glide-engineer

### 분석 요청 시

1. 분석 대상 파악 (Glide / PB / 둘 다)
2. 해당 analyst agent(s) 호출
3. 통합 분석이면 두 analyst 결과를 종합하여 교차 인사이트 제공

### 질문 응답 시

1. **질문 분류**: Glide pipeline, PB pipeline, 메트릭 해석, Schrodinger 도구, 통합 비교 중 어느 영역
2. **근거 확인**: 해당 references 파일 참조
3. **코드 확인**: 실제 코드를 읽어 references와 일치하는지 검증
4. **답변 원칙**:
   - 코드를 근거로: 실제 코드를 읽고 답한다
   - 대안도 함께: 선택하지 않은 대안과 그 이유도 설명
   - 제약사항 솔직히: 현재 구현의 한계를 숨기지 않는다
   - "왜" 중심: 무엇보다 왜를 설명

## 공통 전처리

두 평가 모두 CIF → protein PDB + ligand SDF 변환이 필요하다. `allatom_design/eval/glide/preprocessing.py`의 기존 함수를 재활용한다:
- `preprocess_structure()`: CIF → protein PDB + ligand SDF + centroid
- `get_protein_pn_unit_iids()` / `get_ligand_pn_unit_iids()`: chain 자동 감지
- `write_ligand_sdf()`: AtomArray → RDKit Mol → SDF
- PoseBusters는 이 전처리 결과(protein.pdb, ligand.sdf)를 직접 입력으로 받음

## 데이터 흐름

```
AF3 CIF + JSON
    |
[공통 전처리] CIF → protein PDB + ligand SDF (preprocessing.py)
    |
    ├─── [Glide pipeline]
    │    ├── protein prep (PrepWizard) → .mae
    │    ├── grid gen → .zip
    │    ├── in-place scoring (mininplace) → CSV + SDF
    │    ├── re-docking (confgen) → CSV + SDF
    │    └── RMSD vs reference
    │
    └─── [PoseBusters]
         ├── AF3 prediction: bust(ligand.sdf, receptor.pdb)
         ├── Mininplace output: bust(mininplace.sdf, receptor.pdb) [optional]
         └── Redocking output: bust(redock.sdf, receptor.pdb) [optional]
    |
[통합] Glide metrics + PB metrics → unified CSV
```

## 레퍼런스

작업 시 아래 레퍼런스를 상황에 맞게 참조한다. 모든 레퍼런스를 항상 읽을 필요는 없다 — 해당 영역의 작업/질문일 때만 로드한다.

### Glide 관련
| 문서 | 위치 | 로드 시점 |
|------|------|----------|
| 구현 결정 근거 | `glide-eval/references/implementation-decisions.md` | Glide 설계 의도 질문, 코드 수정 시 |
| 코드 맵 | `glide-eval/references/codebase-pointers.md` | 코드 위치 확인, 새 기능 추가 시 |
| 환경 설정 | `glide-eval/references/environment-config.md` | 경로/환경 관련 문제 시 |

### PoseBusters 관련
| 문서 | 위치 | 로드 시점 |
|------|------|----------|
| API 가이드 | `posebusters-eval/references/api-guide.md` | PB 코드 작성, API 사용법 질문 시 |
| 코드 맵 | `posebusters-eval/references/codebase-pointers.md` | 코드 위치 확인, 새 기능 추가 시 |
| 메트릭 가이드 | `posebusters-eval/references/metric-guide.md` | PB 결과 분석, 메트릭 해석 질문 시 |

레퍼런스 경로는 `.claude/skills/` 기준 상대 경로이다.

## 소스 코드 위치

### Glide pipeline
| 파일 | 역할 |
|------|------|
| `allatom_design/eval/glide/preprocessing.py` | CIF → protein PDB + ligand SDF (공통 전처리) |
| `allatom_design/eval/glide/schrodinger_runner.py` | PrepWizard, Grid Gen, LigPrep, Glide subprocess |
| `allatom_design/eval/glide/result_parser.py` | Glide CSV/SDF 출력 파싱 |
| `allatom_design/eval/glide/pipeline.py` | 단일 샘플 평가 오케스트레이션 |
| `allatom_design/eval/glide/sample_selection.py` | AF3 metrics 기반 샘플 선택 |
| `allatom_design/eval/glide/run_glide_eval_batch.py` | 배치 평가 진입점 |

### PoseBusters
| 파일 | 역할 |
|------|------|
| `allatom_design/eval/eval_utils/eval_posebusters.py` | PoseBusters evaluation 핵심 로직 |
| `allatom_design/eval/eval_utils/eval_metrics.py` | 기존 docking metrics (RMSD 등) |

### 테스트
| 파일 | 역할 |
|------|------|
| `allatom_design/tests/glide/` | Glide 파이프라인 테스트 (47개) |

## 환경 요구사항

- Python 환경: lullaby_local conda 환경 (biotite 1.5.0, rdkit, posebusters 0.6.0)
- Schrodinger: config로 관리되는 경로 (`glide-eval/references/environment-config.md` 참조)
- 기존 환경의 패키지를 제거하거나 버전을 변경하지 않는다
- Schrodinger Python API가 필요한 작업은 `$SCHRODINGER/run`으로 별도 프로세스 실행

## 에러 핸들링

| 상황 | 전략 |
|------|------|
| Schrodinger 도구 실패 | stderr 로깅, 해당 sample skip, PB는 계속 실행 |
| PoseBusters bust() 실패 | 해당 sample skip, 로그 기록, Glide 결과는 유지 |
| CIF → PDB/SDF 변환 실패 | 에러 로깅, 대안 시도. 전처리 실패 시 Glide/PB 모두 skip |
| Reference structure 없음 | RMSD 비교 skip, "dock" config fallback |
| Batch 중 일부 실패 | 성공한 sample로 결과 생성, 실패 목록 별도 저장 |
| 에이전트 1개 실패 | 1회 재시도. 재실패 시 해당 결과 없이 진행, 보고서에 누락 명시 |

## 테스트 시나리오

### 정상 흐름
1. example_data/의 0H7 sample로 전체 파이프라인 실행 요청
2. Phase 1: Glide + PB 둘 다 필요 → 환경 확인
3. Phase 2: glide-engineer가 Glide pipeline 실행 → in-place GlideScore 산출
4. Phase 2: pb-engineer가 PB bust() 실행 → validity check 결과
5. Phase 3: 두 analyst 병렬 호출 → Glide 해석 + PB 해석
6. 통합: GlideScore + pb_valid 포함 unified CSV 생성
7. 교차 인사이트: "pb_valid=True이고 GlideScore=-7.2 → 화학적으로 valid하고 에너지적으로도 안정적"

### 에러 흐름
1. Schrodinger 경로가 잘못 설정된 상태에서 전체 파이프라인 요청
2. Phase 2: glide-engineer가 Schrodinger 에러 감지 → Glide skip
3. Phase 2: pb-engineer는 정상 실행 (Schrodinger 불필요) → PB 결과 생성
4. Phase 3: pb-analyst만 호출, Glide 결과 없음을 명시
5. 사용자에게 "Glide 실패 (Schrodinger 경로 확인 필요), PB 결과만 보고" 안내
