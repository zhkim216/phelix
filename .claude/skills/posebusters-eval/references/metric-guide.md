# PoseBusters Metric Guide

PoseBusters가 체크하는 각 메트릭의 의미, 카테고리, 해석 방법.

## 목차
1. [Summary metric: pb_valid](#summary-metric-pb_valid)
2. [Chemical validity checks](#chemical-validity-checks)
3. [Intramolecular validity checks](#intramolecular-validity-checks)
4. [Intermolecular validity checks](#intermolecular-validity-checks)
5. [해석 가이드라인](#해석-가이드라인)
6. [자주 실패하는 패턴](#자주-실패하는-패턴)

---

## Summary metric: pb_valid

```python
pb_valid = (모든 validity check가 True)
```

PoseX는 RMSD 컬럼을 제외한 나머지 모든 check를 AND 연산한다. **단 하나라도 False면 pb_valid=False.**

가장 중요한 단일 지표. 논문에서 "PB Valid" rate로 보고됨.

---

## Chemical validity checks

분자의 화학 구조 자체가 올바른지 검증.

| 메트릭 | 의미 | 실패 원인 |
|--------|------|----------|
| `mol_pred_loaded` | predicted 분자가 로드 가능한지 | 잘못된 SDF, atom typing 에러 |
| `mol_true_loaded` | reference 분자가 로드 가능한지 (redock only) | reference SDF 문제 |
| `mol_cond_loaded` | receptor(protein)이 로드 가능한지 | CIF 형식 사용 (PDB 필요), 잘못된 PDB |
| `sanitization` | RDKit sanitization 통과 | 비정상적 원자가, 잘못된 formal charge |
| `molecular_formula` | predicted vs reference 화학식 일치 (redock only) | 원자 추가/삭제, 잘못된 atom type assignment |
| `molecular_bonds` | 결합 구조 일치 (redock only) | bond order 변경, ring opening/closing |
| `tetrahedral_chirality` | chirality 보존 (redock only) | chiral center 반전 |
| `double_bond_stereochemistry` | E/Z stereochemistry 보존 (redock only) | cis-trans isomerism 변경 |

> **"dock" config에서는** mol_true_loaded, molecular_formula, molecular_bonds, tetrahedral_chirality, double_bond_stereochemistry가 체크되지 않는다.

---

## Intramolecular validity checks

분자 내부의 기하학적 합리성 검증.

| 메트릭 | 의미 | 실패 원인 | 해석 |
|--------|------|----------|------|
| `bond_lengths` | 결합 길이가 CSD (Cambridge Structural Database) 통계 범위 내 | 비정상적으로 늘어나거나 줄어든 결합 | generative model이 원자 위치를 부정확하게 배치 |
| `bond_angles` | 결합 각도가 CSD 통계 범위 내 | 비정상적 결합 각 | geometry optimization 부족 |
| `internal_steric_clash` | 분자 내 원자 간 비정상적 근접 없음 | 분자 내 원자가 VDW 반지름보다 가까움 | 접힌 구조, 고에너지 conformation |
| `aromatic_ring_flatness` | 방향족 고리가 평면적 | 방향족 고리가 구부러짐 | AF3가 ring geometry를 잘못 예측 |
| `double_bond_flatness` | 이중결합 주위가 평면적 | 이중결합 주위 원자가 같은 평면에 있지 않음 | 이중결합 geometry 오류 |
| `internal_energy` | 분자 내부 에너지가 합리적 범위 | 비정상적으로 높은 internal energy | 고에너지 conformation, strain |

---

## Intermolecular validity checks

protein-ligand 상호작용의 합리성 검증.

| 메트릭 | 의미 | 실패 원인 | 해석 |
|--------|------|----------|------|
| `minimum_distance_to_protein` | ligand-protein 최소 거리 > threshold | 원자가 겹침 (steric clash) | ligand가 protein backbone/sidechain에 박힘 |
| `minimum_distance_to_organic_cofactors` | ligand-cofactor 최소 거리 > threshold | cofactor와 겹침 | cofactor 위치 인식 실패 |
| `minimum_distance_to_inorganic_cofactors` | ligand-inorganic 최소 거리 > threshold | 금속 이온과 겹침 | 금속 coordination 오류 |
| `volume_overlap_with_protein` | ligand-protein volume overlap 없음 | 상당한 volume overlap | ligand가 protein cavity 밖으로 나오거나 벽에 묻힘 |
| `volume_overlap_with_organic_cofactors` | ligand-cofactor volume overlap 없음 | cofactor와 공간 충돌 | 다중 ligand binding site 인식 실패 |
| `volume_overlap_with_inorganic_cofactors` | ligand-inorganic volume overlap 없음 | 금속과 공간 충돌 | 금속 coordination geometry 오류 |

> plinder에서는 추가로 `minimum_distance_to_waters`, `volume_overlap_with_waters`도 체크. 버전에 따라 체크 항목이 다를 수 있다.

---

## 해석 가이드라인

### pb_valid 비율 기준값
- **> 80%**: 우수. 대부분의 predicted pose가 화학적/물리적으로 합리적.
- **50-80%**: 보통. 일부 pose에 문제가 있으나 전반적으로 양호.
- **< 50%**: 문제 있음. 어떤 카테고리에서 실패하는지 분석 필요.

### 카테고리별 의미
- **Chemical validity 실패 많음**: atom type assignment이나 분자 변환 파이프라인에 문제.
- **Intramolecular 실패 많음**: generative model이 분자 내부 geometry를 잘못 예측. bond lengths와 ring flatness가 가장 흔한 실패.
- **Intermolecular 실패 많음**: ligand가 protein과 비현실적으로 상호작용. steric clash가 주원인.

### 비교 분석 시
- **AF3 vs mininplace**: mininplace에서 pb_valid가 올라가면, Glide minimization이 pose quality를 개선함을 의미.
- **AF3 vs redocking**: redocking에서 pb_valid가 높으면, receptor structure는 좋지만 AF3 pose에 문제.
- **RMSD 낮은데 pb_valid False**: chemical validity 문제 (bond 꼬임, chirality 반전 등). geometry는 맞지만 화학이 틀린 케이스.

---

## 자주 실패하는 패턴

1. **bond_lengths 실패**: generative model에서 가장 흔한 실패. 특히 금속 coordination bond나 특이한 functional group.
2. **internal_steric_clash**: 큰 유연한 ligand에서 자주 발생. ring이나 aliphatic chain이 접힌 conformation.
3. **volume_overlap_with_protein**: AF3가 ligand를 protein surface 바로 안쪽에 배치할 때 발생.
4. **aromatic_ring_flatness**: AF3의 ring geometry prediction이 부정확할 때. 특히 fused ring systems.
5. **mol_pred_loaded 실패**: SDF 변환 파이프라인 (atom_array_to_rdkit → MolToMolFile) 에 문제가 있을 때. SMILES template 없이 변환하면 bond order가 틀릴 수 있다.
