#!/usr/bin/env bash
# [폐쇄망] 컨테이너 실행 래퍼. install_bundle.sh로 설치했으면 'mcc' 명령으로도 부를 수 있다. 예)
#   bash docker/run.sh train
#   bash docker/run.sh predict --input /data/inbox/a.xlsx --output /work/out/a_pred.xlsx
#   bash docker/run.sh evaluate --input /data/eval/foreign_labeled.xlsx
#   bash docker/run.sh batch --inbox /data/inbox --outbox /work/out
# 설정 파일을 바꾸려면 CONFIG=/app/configs/site.yaml, 경로는 DATA_DIR/MODELS_DIR/WORK_DIR로 바꾼다.
# (선택) 번역 증강에 LiteLLM을 쓰면 키를 환경 변수로 넘긴다: LITELLM_API_KEY=sk-... bash docker/run.sh translate ...
# LiteLLM이 같은 서버에서 돌고 있으면 DOCKER_NETWORK=host 로 실행하고 base_url을 http://localhost:4000/v1 로 둔다.
# 이 저장소의 mcc 코드를 컨테이너에 덮어 마운트하므로, 의존성이 바뀌지 않는 한 코드 ZIP만 새로 반입하면 된다.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

IMAGE=${IMAGE:-mcc:latest}
DATA_DIR=${DATA_DIR:-/srv/mcc/data}
MODELS_DIR=${MODELS_DIR:-/srv/mcc/models}
WORK_DIR=${WORK_DIR:-/srv/mcc/work}
CONFIG=${CONFIG:-/app/configs/default.yaml}

# GPU는 기본으로 0번 한 장만 쓴다(공용 서버에서 다른 팀 GPU를 건드리지 않게). GPUS=device=1 / all / none 으로 바꿀 수 있다.
# nvidia-smi가 없으면(GPU 없는 검증용 PC) --gpus를 빼고 CPU로 실행한다. WSL2에서는 nvidia-smi가 /usr/lib/wsl/lib에 있다.
if [ -z "${GPUS+x}" ]; then
  smi=$(command -v nvidia-smi || true); [ -n "$smi" ] || [ ! -x /usr/lib/wsl/lib/nvidia-smi ] || smi=/usr/lib/wsl/lib/nvidia-smi
  if [ -n "$smi" ] && "$smi" -L >/dev/null 2>&1; then GPUS=device=0; else GPUS=none; echo "[run.sh] GPU 없음: CPU로 실행" >&2; fi
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
