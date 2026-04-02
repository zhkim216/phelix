# PoseBusters API Guide

PoseBusters API 사용법과 PoseX/plinder의 구현 패턴. 코드 작성 시 반드시 참조한다.

## 목차
1. [기본 API](#기본-api)
2. [Config 모드](#config-모드)
3. [입력 형식](#입력-형식)
4. [출력 형식과 결과 파싱](#출력-형식과-결과-파싱)
5. [PoseX 구현 패턴](#posex-구현-패턴)
6. [plinder 구현 패턴](#plinder-구현-패턴)
7. [주의사항](#주의사항)

---

## 기본 API

```python
from posebusters import PoseBusters

# 단일 entry 평가
pb = PoseBusters(config="dock")  # 또는 "redock"
result = pb.bust(
    mol_pred="predicted_ligand.sdf",
    mol_true="reference_ligand.sdf",  # redock일 때만 필요
    mol_cond="receptor.pdb",
    full_report=True,
)
# result: pandas DataFrame

# 배치 평가
pb = PoseBusters(config="redock", top_n=None, max_workers=8)
bust_data = pd.DataFrame({
    "mol_pred": [...],  # predicted ligand SDF 경로 리스트
    "mol_true": [...],  # reference ligand SDF 경로 리스트
    "mol_cond": [...],  # receptor PDB 경로 리스트
})
results = pb.bust_table(bust_data, full_report=True)
# results: pandas DataFrame (모든 entry 결합)
```

### bust() vs bust_table()
- `bust()`: 단일 entry. 경로 문자열 또는 RDKit Mol 객체를 직접 전달.
- `bust_table()`: 배치. DataFrame에 경로를 담아 전달. `max_workers`로 병렬 처리.
- 대량 처리 시 `bust_table()`이 효율적.

### full_report 파라미터
- `full_report=True`: 개별 sub-test 결과를 모두 포함 (bond_lengths, bond_angles 등)
- `full_report=False`: 요약 결과만 (각 카테고리별 pass/fail)
- 분석용으로는 `True` 권장.

---

## Config 모드

### "dock" (reference 없음)
- mol_true 불필요
- predicted pose의 화학적/물리적 validity만 체크
- 체크 항목: 분자 로딩, sanitization, bond lengths, bond angles, steric clash, ring flatness, internal energy, protein과의 거리/overlap

### "redock" (reference 있음)
- mol_true 필요
- "dock"의 모든 체크 + 추가 검증:
  - `molecular_formula`: predicted vs reference 화학식 일치
  - `molecular_bonds`: 결합 구조 일치
  - `tetrahedral_chirality`: chirality 보존
  - `double_bond_stereochemistry`: stereochemistry 보존
  - `rmsd_≤_2å`: symmetry-corrected RMSD (PoseX에서 사용)
- reference ligand이 있으면 "redock"이 더 엄격한 검증을 제공

### 선택 기준
- AF3 prediction에서 designed structure의 ligand이 reference로 있으면 → "redock" 가능
- Reference 없이 predicted pose만 검증하고 싶으면 → "dock"
- RMSD는 이미 eval_metrics.py에서 계산하므로, PB에서 RMSD를 중복 계산할 필요는 없음
- **실용적 추천**: "dock" config로 validity만 체크하고, RMSD는 기존 코드 사용

---

## 입력 형식

### mol_pred (predicted ligand)
- **형식**: SDF 파일 경로 또는 RDKit Mol 객체
- **요구사항**: 3D coordinates 포함, hydrogen 있어도 없어도 됨
- **출처**: AF3 output CIF에서 ligand 추출 → SDF 변환 / Glide output SDF

### mol_true (reference ligand, redock 전용)
- **형식**: SDF 파일 경로 또는 RDKit Mol 객체
- **요구사항**: 3D coordinates 포함
- **출처**: designed structure에서 ligand 추출 → SDF

### mol_cond (receptor/conditioning molecule)
- **형식**: PDB 파일 경로 (CIF 불가!)
- **요구사항**: protein structure, ligand 제외
- **출처**: AF3 output CIF에서 protein 추출 → PDB 변환

### CIF → PDB/SDF 변환
기존 Glide preprocessing 코드를 재활용:

```python
# allatom_design/eval/glide/preprocessing.py 패턴
from allatom_design.utils.sample_io_utils import load_example_with_parse
from atomworks.io.utils.io_utils import to_pdb_string
from atomworks.io.tools.rdkit import atom_array_to_rdkit
import atomworks.enums as aw_enums

# CIF 로드
atom_array, _ = load_example_with_parse(cif_path)

# protein 추출 → PDB
protein_mask = atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L
protein_atoms = atom_array[protein_mask]
pdb_string = to_pdb_string(protein_atoms)

# ligand 추출 → SDF
ligand_mask = atom_array.chain_type == aw_enums.ChainType.NON_POLYMER
ligand_atoms = atom_array[ligand_mask]
rdkit_mol = atom_array_to_rdkit(ligand_atoms, smiles=ligand_smiles)
Chem.MolToMolFile(rdkit_mol, sdf_path)
```

---

## 출력 형식과 결과 파싱

### bust() 반환값
pandas DataFrame. 컬럼은 test 이름, 행은 각 molecule.

### 결과 키 파싱 (plinder에서 발견된 quirk)
bust() 결과의 dict/DataFrame 인덱스가 `(sdf_path, mol_name)` tuple일 수 있다:

```python
# plinder 패턴 (utils.py:368-392)
result_dict = pb.bust(...).to_dict()

# key가 (sdf_path, mol_name) tuple
# mol_name은 SDF 파일 내 이름 또는 기본값 "mol_at_pos_0"
for key, value in result_dict.items():
    if isinstance(key, tuple):
        sdf_path, mol_name = key
```

**bust_table()은 이 문제가 덜하다** — DataFrame으로 깔끔하게 반환.

### pb_valid 계산
```python
# PoseX 패턴 (calculate_benchmark_result.py:101-102)
test_columns = bust_results[POSEBUSTER_TEST_COLUMNS].copy()
bust_results["pb_valid"] = test_columns.iloc[:, 1:].all(axis=1)
# RMSD 컬럼(index 0)을 제외한 모든 validity check가 True이면 pb_valid=True
```

---

## PoseX 구현 패턴

**파일**: `/home/possu/jinho/PoseX/scripts/calculate_benchmark_result.py`

### 핵심 코드 흐름

```python
# 1. PoseBusters 초기화
buster = PoseBusters(config="redock", top_n=None, max_workers=args.max_workers)
buster.config["loading"]["mol_true"]["load_all"] = True

# 2. 입력 DataFrame 구성
bust_data = pd.DataFrame({
    "mol_pred": aligned_ligand_sdf_paths,
    "mol_true": reference_ligand_sdf_paths,
    "mol_cond": aligned_protein_pdb_paths,
})

# 3. bust_table() 실행
bust_results = buster.bust_table(bust_data, full_report=True)

# 4. 결과 처리
test_data = bust_results[POSEBUSTER_TEST_COLUMNS].copy()
bust_results.loc[:, "pb_valid"] = test_data.iloc[:, 1:].all(axis=1)
```

### PoseX의 test columns 전체 목록
```python
POSEBUSTER_TEST_COLUMNS = [
    "rmsd_≤_2å",
    # chemical validity
    "mol_pred_loaded", "mol_true_loaded", "mol_cond_loaded",
    "sanitization", "molecular_formula", "molecular_bonds",
    "tetrahedral_chirality", "double_bond_stereochemistry",
    # intramolecular
    "bond_lengths", "bond_angles", "internal_steric_clash",
    "aromatic_ring_flatness", "double_bond_flatness", "internal_energy",
    # intermolecular
    "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
]
```

### PoseX의 최종 메트릭
- `RMSD <= 2Å`: RMSD 기준 성공률
- `RMSD <= 2Å AND PB Valid`: RMSD + 화학/물리 validity 모두 통과 비율

---

## plinder 구현 패턴

**파일**: `/home/possu/jinho/allatom-design/plinder/src/plinder/eval/docking/utils.py`

### 핵심 코드 흐름

```python
# 1. PoseBusters 초기화 (dock config — reference 비교 포함이지만 별도 RMSD 계산)
pb = PoseBusters(config="dock")

# 2. 단일 entry bust()
result_dict = pb.bust(
    mol_pred=ligand_class.sdf_file,
    mol_true=ligand_class.reference_ligand[self.posebusters_mapper].sdf_file,
    mol_cond=self.model.receptor_file,  # 반드시 PDB!
    full_report=self.score_posebusters_full_report,
).to_dict()

# 3. 결과 파싱 (키가 tuple일 수 있음)
# bust() 결과 키: (sdf_file_path, molecule_name) 또는 단순 문자열
# plinder는 3가지 키 형식을 순서대로 시도:
#   1) OST-prefixed name (e.g., "00001_1.D")
#   2) "mol_at_pos_0" (PB 기본값)
#   3) SDF 파일 stem

# 4. 결과 집계 (atom-count weighted average)
# 여러 ligand가 있을 때 heavy atom 수 기반 가중 평균
```

### plinder의 PB 메트릭 목록 (posebusters_ 접두사)
```
posebusters_mol_pred_loaded
posebusters_mol_cond_loaded
posebusters_sanitization
posebusters_inchi_convertible
posebusters_all_atoms_connected
posebusters_bond_lengths
posebusters_bond_angles
posebusters_internal_steric_clash
posebusters_aromatic_ring_flatness
posebusters_double_bond_flatness
posebusters_internal_energy
posebusters_protein-ligand_maximum_distance
posebusters_minimum_distance_to_protein
posebusters_minimum_distance_to_organic_cofactors
posebusters_minimum_distance_to_inorganic_cofactors
posebusters_minimum_distance_to_waters
posebusters_volume_overlap_with_protein
posebusters_volume_overlap_with_organic_cofactors
posebusters_volume_overlap_with_inorganic_cofactors
posebusters_volume_overlap_with_waters
```

### plinder의 CIF 제약 처리

```python
# plinder/eval/docking/utils.py:354-359
# PoseBusters는 .cif receptor를 읽지 못함
if not str(self.model.receptor_file).endswith(".pdb"):
    logger.warning(
        f"PoseBusters may not load receptor file: {self.model.receptor_file}"
    )
```

plinder에서는 `receptor_pdb` property를 사용하여 PDB 형식을 보장한다.

---

## 주의사항

### 1. CIF receptor 불가
PoseBusters는 내부적으로 RDKit으로 receptor를 읽는데, RDKit이 CIF를 지원하지 않는다. 반드시 PDB로 변환해야 한다.

### 2. SDF 파일 내 molecule name
bust() 결과의 키가 SDF 파일 내 molecule name에 의존한다. `atom_array_to_rdkit()` 후 `Chem.MolToMolFile()`로 저장할 때, mol.SetProp("_Name", name)을 명시적으로 설정하면 결과 파싱이 수월해진다.

### 3. Hydrogen 처리
PoseBusters가 자체적으로 hydrogen을 추가/제거하므로, 입력에 hydrogen이 있든 없든 결과는 동일하다.

### 4. 버전 차이
- PoseX: posebusters 0.4.4
- plinder: posebusters 0.3.1
- lullaby_local: posebusters 0.6.0
- 버전에 따라 test column 이름이나 개수가 다를 수 있다. 0.6.0의 실제 출력 컬럼을 확인하고 사용한다.

### 5. 성능
bust()는 entry당 수 초 소요. 대량 처리 시 bust_table()의 max_workers를 활용한다.

### 6. multi-ligand 처리
한 complex에 여러 ligand가 있으면 각 ligand별로 bust()를 호출하고, 결과를 집계한다. plinder는 atom-count weighted average를 사용한다.
