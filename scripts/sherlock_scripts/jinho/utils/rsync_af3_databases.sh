#!/bin/bash
#SBATCH --job-name=rsync_af3_db
#SBATCH -p bioe,possu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-0:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/rsync_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/rsync_%j.err

## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/zhkim216/slurm_log_files/job_$SLURM_JOB_ID.sh

source ~/.bashrc

# 시작 시간 기록
echo "==================================="
echo "Rsync job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "==================================="

# 소스와 타겟 디렉토리 정의
SOURCE_DIR="/scratch/groups/possu/af3_resources/databases"
TARGET_DIR="/scratch/users/zhkim216/af3_databases"

# 타겟 디렉토리 생성
mkdir -p $TARGET_DIR

# 디스크 공간 확인
echo "Checking disk space..."
echo "Source size:"
du -sh $SOURCE_DIR
echo "Target available space:"
df -h $TARGET_DIR

# 옵션 3: 단순 rsync (병렬 처리 없이, 하지만 더 안정적)
rsync -avP --no-links $SOURCE_DIR/ $TARGET_DIR/

# 검증: 파일 개수 비교
echo "==================================="
echo "Verification:"
echo "Source file count:"
find $SOURCE_DIR -type f | wc -l
echo "Target file count:"
find $TARGET_DIR -type f | wc -l

# 크기 비교
echo "Source total size:"
du -sh $SOURCE_DIR
echo "Target total size:"
du -sh $TARGET_DIR

# 종료 시간 기록
echo "==================================="
echo "Rsync job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "===================================