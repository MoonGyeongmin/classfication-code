#!/usr/bin/env bash
# [인터넷 PC] 이미지 빌드 + 모델 다운로드 후 반입용 파일을 dist/에 만든다.
#   bash docker/build_offline.sh            # 이미지 + 인코더 + 번역 LLM
#   SKIP_LLM=1 bash docker/build_offline.sh # 번역 LLM 제외
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=${IMAGE:-mcc:0.1.0}
ENCODER=${ENCODER:-BAAI/bge-m3}
LLM=${LLM:-Qwen/Qwen2.5-32B-Instruct}
DIST=${DIST:-dist}
mkdir -p "$DIST/models"

docker build -f docker/Dockerfile ${BASE_IMAGE:+--build-arg BASE_IMAGE="$BASE_IMAGE"} -t "$IMAGE" .
docker save "$IMAGE" | gzip > "$DIST/${IMAGE/:/_}.tar.gz"

download() {  # $1: HF 저장소, $2: 저장 폴더명
  docker run --rm -e HF_HUB_OFFLINE=0 -v "$PWD/$DIST/models:/models" --entrypoint python "$IMAGE" -c \
    "from huggingface_hub import snapshot_download as d; d('$1', local_dir='/models/$2', allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.py','tokenizer*'])"
}
download "$ENCODER" "$(basename "$ENCODER")"
[ "${SKIP_LLM:-0}" = 1 ] || download "$LLM" "$(basename "$LLM")"

tar -C "$DIST" -cf "$DIST/models.tar" models
( cd "$DIST" && sha256sum ./*.tar.gz models.tar > SHA256SUMS )
echo "반입할 파일:"; ls -lh "$DIST"/*.tar.gz "$DIST"/models.tar "$DIST"/SHA256SUMS
