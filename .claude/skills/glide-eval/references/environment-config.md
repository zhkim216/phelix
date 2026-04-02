# Environment Configuration

## 환경별 경로

### Local
- Schrodinger: /home/possu/jinho/software/schrodinger
- 프로젝트: /home/possu/jinho/allatom-design
- Reference (denovo): /home/possu/jinho/datasets/val_cifs/denovo_val_cifs
- Reference (native): /home/possu/jinho/datasets/val_cifs/native_val_cifs
- Conda 환경: lullaby_local
- 작업 디렉토리: /home/possu/jinho/allatom-design/debug/260325_glide_debug

### Sherlock (SLURM)
- Schrodinger: /scratch/users/zhkim216/software/schrodinger2025-3
- 기타 경로: 사용자가 Sherlock 환경에 맞게 config에서 지정

## Config 구조 원칙
- 환경별 차이는 config override로 처리한다
- 기본 config + 환경별 override 패턴
- 절대 경로를 코드에 하드코딩하지 않는다
- Schrodinger 경로는 `$SCHRODINGER` 환경변수 또는 config에서 읽는다

## Python 환경 원칙
- lullaby_local에 설치된 패키지: biotite 1.5.0, rdkit, openbabel, atomworks
- Schrodinger Python API는 `$SCHRODINGER/run`으로 별도 프로세스 실행
- lullaby_local 환경에 새 패키지를 설치하거나 기존 버전을 변경하지 않는다
