---
name: posebusters-eval
description: "[DEPRECATED — ligand-eval로 통합됨] 이 스킬을 직접 트리거하지 말 것. PoseBusters 관련 요청은 ligand-eval 스킬이 처리한다. 이 디렉토리의 references/만 참조용으로 유지."
---

# PoseBusters Evaluation (Archived)

> **이 스킬은 `ligand-eval`로 통합되었다.** PoseBusters 관련 요청은 `ligand-eval` 스킬을 사용한다.
> 이 디렉토리의 `references/` 파일들은 에이전트와 통합 스킬이 참조하므로 유지한다.

아래는 아카이브된 원본 내용이다.

## 원본 설명

AF3 predicted structure 및 Glide docking 결과의 ligand pose가 화학적/물리적으로 valid한지 PoseBusters로 검증한다.

## 배경

PoseBusters는 generated molecule pose의 plausibility를 체크하는 라이브러리다:
- **Chemical validity**: 분자 로딩, sanitization, 화학식/결합 일관성, chirality, stereochemistry
- **Intramolecular validity**: bond lengths, bond angles, steric clash, ring flatness, internal energy
- **Intermolecular validity**: protein과의 최소 거리, volume overlap

현재 eval pipeline에는 SC RMSD, pLDDT, ligand RMSD, Glide scores가 있지만, **ligand의 화학적/물리적 validity check가 없다.** PoseBusters가 이 gap을 채운다.

## 적용 대상 (config flag로 제어)

| 대상 | 입력 | 설명 |
|------|------|------|
| AF3 prediction | AF3 output CIF | AF3가 예측한 ligand pose의 validity |
| Glide mininplace | Glide in-place scoring output SDF | AF3 pose를 Glide로 minimize한 결과 |
| Glide redocking | Glide re-docking output SDF | 같은 receptor에 re-dock한 best pose |

세 대상은 독립적으로 또는 함께 실행 가능하다.

## 설계 원칙

1. **모듈화**: 각 기능(CIF→SDF 변환, bust 실행, 결과 파싱, 메트릭 집계)을 독립 함수로 분리한다. 하나의 함수가 여러 책임을 갖지 않는다.
2. **재활용성**: 기존 코드베이스에 이미 존재하는 전처리·변환 로직(`glide/preprocessing.py` 등)을 최대한 재활용한다. 새 유틸리티를 만들 때도 PoseBusters 전용이 아닌, 다른 eval pipeline에서도 쓸 수 있는 범용 인터페이스로 설계한다.

## 핵심 제약사항

1. **PoseBusters는 CIF를 읽지 못한다** — receptor는 반드시 PDB로 변환해야 한다
2. **PoseBusters 입력 3개**: `mol_pred` (predicted ligand SDF), `mol_true` (reference ligand SDF, optional), `mol_cond` (receptor PDB)
3. **bust() 결과 키가 tuple** — `(sdf_path, mol_name)` 형태. SDF 내 이름이 없으면 `"mol_at_pos_0"` 기본값
4. **"dock" vs "redock"**: reference ligand 없으면 "dock", 있으면 "redock" (더 엄격한 화학 구조 검증)
5. **posebusters 0.6.0** 이 lullaby_local에 설치되어 있음

## 코드 위치

| 파일 | 역할 |
|------|------|
| `allatom_design/eval/eval_utils/eval_posebusters.py` | PoseBusters evaluation 핵심 로직 **(신규 작성)** |
| `allatom_design/eval/eval_utils/eval_metrics.py` | 기존 docking metrics (패턴 참조) |
| `allatom_design/eval/glide/preprocessing.py` | CIF → PDB/SDF 전처리 (재활용) |
| `allatom_design/eval/glide/pipeline.py` | Glide pipeline (통합 대상) |

## 레퍼런스

| 문서 | 위치 | 내용 |
|------|------|------|
| API 가이드 | `references/api-guide.md` | PoseBusters API, 입출력 형식, PoseX/plinder 코드 패턴, 주의사항 |
| 코드베이스 포인터 | `references/codebase-pointers.md` | 재활용할 기존 코드 위치 |
| 메트릭 가이드 | `references/metric-guide.md` | PB 메트릭 목록, 카테고리, 해석 방법 |

## 에이전트 구성

| 에이전트 | subagent_type | 역할 |
|---------|--------------|------|
| pb-engineer | pb-engineer | 코드 구현, 디버깅 |
| pb-analyst | pb-analyst | 결과 분석, 메트릭 해석 |

## 에러 핸들링

| 상황 | 전략 |
|------|------|
| bust() 실패 | 해당 sample skip, 로그 기록 |
| CIF → PDB 변환 실패 | 에러 로깅, 대안 시도 |
| Reference ligand 없음 | "dock" config로 fallback |
| Batch 중 일부 실패 | 성공한 sample로 결과 생성, 실패 목록 별도 저장 |
