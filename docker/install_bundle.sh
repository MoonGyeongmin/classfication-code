#!/usr/bin/env bash
# [폐쇄망] GitHub Release에서 받은 파일을 한 폴더에 모아 반입한 뒤 그 폴더에서 실행한다.
#   bash install_bundle.sh                 # 체크섬 확인 -> 조각 합치기 -> docker load -> 모델 배치
#   MODELS_DIR=/srv/mcc/models bash install_bundle.sh
# 이미 합쳐진 파일은 건너뛰므로 중간에 끊겨도 다시 실행하면 된다.
set -euo pipefail
cd "$(dirname "$0")"
MODELS_DIR=${MODELS_DIR:-/srv/mcc/models}

echo "== 1. 체크섬 확인"
cat SHA256SUMS.* | sha256sum -c --quiet
echo "모든 파일 정상"

echo "== 2. 조각 합치기"
for first in *.part-aa; do
  [ -e "$first" ] || continue
  whole=${first%.part-aa}
  [ -e "$whole" ] && { echo "있음: $whole"; continue; }
  cat "$whole".part-* > "$whole.tmp" && mv "$whole.tmp" "$whole"
  echo "합침: $whole"
done

echo "== 3. 이미지 불러오기"
for img in mcc_*.tar.gz; do docker load -i "$img"; done

echo "== 4. 모델 배치 -> $MODELS_DIR"
for f in models__*; do
  case "$f" in *.part-*|*.tmp) continue ;; esac
  rel=${f#models__}; rel=${rel//__//}
  mkdir -p "$MODELS_DIR/$(dirname "$rel")"
  [ -e "$MODELS_DIR/$rel" ] || cp "$f" "$MODELS_DIR/$rel"
done
ls "$MODELS_DIR"
echo "완료. 조각 파일(*.part-*)은 지워도 된다. 다음: nvidia-smi, bash docker/run.sh --help"
