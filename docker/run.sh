#!/usr/bin/env bash
# [폐쇄망] 컨테이너 실행 래퍼. 예)
#   bash docker/run.sh train
#   bash docker/run.sh translate --input /data/train/fa50_ko.xlsx --output /data/train/fa50_aug.xlsx
#   bash docker/run.sh predict --input /data/inbox/a.xlsx --output /work/out/a_pred.xlsx
#   bash docker/run.sh batch --inbox /data/inbox --outbox /work/out
# 설정 파일을 바꾸려면 CONFIG=/app/configs/site.yaml, 경로는 DATA_DIR/MODELS_DIR/WORK_DIR로 바꾼다.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=${IMAGE:-mcc:0.1.0}
DATA_DIR=${DATA_DIR:-/srv/mcc/data}
MODELS_DIR=${MODELS_DIR:-/srv/mcc/models}
WORK_DIR=${WORK_DIR:-/srv/mcc/work}
CONFIG=${CONFIG:-/app/configs/default.yaml}

exec docker run --rm --gpus all --shm-size 16g \
  -v "$DATA_DIR:/data" -v "$MODELS_DIR:/models:ro" -v "$WORK_DIR:/work" \
  -v "$PWD/configs:/app/configs:ro" -v "$PWD/codebook:/app/codebook:ro" \
  "$IMAGE" --config "$CONFIG" "$@"
