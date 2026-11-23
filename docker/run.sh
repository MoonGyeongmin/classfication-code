#!/usr/bin/env bash
# [폐쇄망] 컨테이너 실행 래퍼. install_bundle.sh로 설치했으면 'mcc' 명령으로도 부를 수 있다. 예)
#   bash docker/run.sh train
#   bash docker/run.sh translate --input /data/train/fa50_ko.xlsx --output /data/train/fa50_aug.xlsx
#   bash docker/run.sh predict --input /data/inbox/a.xlsx --output /work/out/a_pred.xlsx
#   bash docker/run.sh batch --inbox /data/inbox --outbox /work/out
# 설정 파일을 바꾸려면 CONFIG=/app/configs/site.yaml, 경로는 DATA_DIR/MODELS_DIR/WORK_DIR로 바꾼다.
# LiteLLM을 쓰면 키를 환경 변수로 넘긴다: LITELLM_API_KEY=sk-... bash docker/run.sh translate ...
# LiteLLM이 같은 서버에서 돌고 있으면 DOCKER_NETWORK=host 로 실행하고 base_url을 http://localhost:4000/v1 로 둔다.
# 이 저장소의 mcc 코드를 컨테이너에 덮어 마운트하므로, 의존성이 바뀌지 않는 한 코드 ZIP만 새로 반입하면 된다.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

IMAGE=${IMAGE:-mcc:latest}
DATA_DIR=${DATA_DIR:-/srv/mcc/data}
MODELS_DIR=${MODELS_DIR:-/srv/mcc/models}
WORK_DIR=${WORK_DIR:-/srv/mcc/work}
CONFIG=${CONFIG:-/app/configs/default.yaml}

# GPU가 없는 PC(검증용)에서도 돌도록 nvidia-smi가 없으면 --gpus를 빼고 CPU로 실행한다. GPUS=all/none 으로 강제 가능.
if [ -z "${GPUS+x}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then GPUS=all; else GPUS=none; echo "[run.sh] GPU 없음: CPU로 실행" >&2; fi
fi
GPU_ARGS=(); [ "$GPUS" = none ] || GPU_ARGS=(--gpus "$GPUS")
# 리허설은 기본으로 네트워크를 끊고(폐쇄망과 같게) 돌린다.
[ "${1:-}" = rehearse ] && DOCKER_NETWORK=${DOCKER_NETWORK:-none}
[ -t 1 ] && TTY_ARGS=(-t) || TTY_ARGS=()

exec docker run --rm "${GPU_ARGS[@]}" "${TTY_ARGS[@]}" --shm-size 16g ${DOCKER_NETWORK:+--network "$DOCKER_NETWORK"} \
  ${LITELLM_API_KEY:+-e LITELLM_API_KEY} -e MCC_HOST_WORK="$WORK_DIR" \
  -v "$DATA_DIR:/data" -v "$MODELS_DIR:/models:ro" -v "$WORK_DIR:/work" \
  -v "$PWD/mcc:/app/mcc:ro" -v "$PWD/configs:/app/configs:ro" -v "$PWD/codebook:/app/codebook:ro" \
  -v "$PWD/tests:/app/tests:ro" -v "$PWD/scripts:/app/scripts:ro" \
  "$IMAGE" --config "$CONFIG" "$@"
