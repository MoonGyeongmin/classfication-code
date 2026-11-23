#!/usr/bin/env bash
# [폐쇄망] GitHub Release에서 받은 파일을 한 폴더에 모아 반입한 뒤 그 폴더에서 실행한다.
#   sudo bash install_bundle.sh
# 체크섬 확인 -> 조각 합치기 -> docker load -> 모델 배치 -> 코드 설치(/srv/mcc/app) -> 'mcc' 명령 등록.
# 이미 끝난 단계는 건너뛰므로 중간에 끊겨도 다시 실행하면 된다.
# 경로를 바꾸려면 MCC_ROOT=/data/mcc 처럼 앞에 붙인다(models, data, work, app이 그 아래에 생긴다).
set -euo pipefail
cd "$(dirname "$0")"
MCC_ROOT=${MCC_ROOT:-/srv/mcc}
MODELS_DIR=${MODELS_DIR:-$MCC_ROOT/models}
APP_DIR=${APP_DIR:-$MCC_ROOT/app}

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
  [ -e "$f" ] || continue
  case "$f" in *.part-*|*.tmp) continue ;; esac
  rel=${f#models__}; rel=${rel//__//}
  mkdir -p "$MODELS_DIR/$(dirname "$rel")"
  [ -e "$MODELS_DIR/$rel" ] || cp "$f" "$MODELS_DIR/$rel"
done
ls "$MODELS_DIR"

if [ -e mcc-app.tar.gz ]; then
  echo "== 5. 코드 설치 -> $APP_DIR"
  tmp=$(mktemp -d)
  tar -C "$tmp" -xzf mcc-app.tar.gz
  if [ -d "$APP_DIR/configs" ]; then  # 다시 설치할 때 고쳐 둔 설정은 그대로 둔다
    rm -rf "$tmp/mcc-app/configs" && cp -a "$APP_DIR/configs" "$tmp/mcc-app/configs"
    echo "기존 설정 유지: $APP_DIR/configs"
  fi
  rm -rf "$APP_DIR" && mkdir -p "$(dirname "$APP_DIR")" && mv "$tmp/mcc-app" "$APP_DIR" && rm -rf "$tmp"
  chmod +x "$APP_DIR/docker/run.sh"
  mkdir -p "$MCC_ROOT/data/inbox" "$MCC_ROOT/data/train" "$MCC_ROOT/work/out"

  echo "== 6. 'mcc' 명령 등록"
  wrapper="#!/usr/bin/env bash
export DATA_DIR=\${DATA_DIR:-$MCC_ROOT/data} MODELS_DIR=\${MODELS_DIR:-$MODELS_DIR} WORK_DIR=\${WORK_DIR:-$MCC_ROOT/work}
exec \"$APP_DIR/docker/run.sh\" \"\$@\""
  if [ -w /usr/local/bin ]; then
    printf '%s\n' "$wrapper" > /usr/local/bin/mcc && chmod +x /usr/local/bin/mcc && echo "등록: /usr/local/bin/mcc"
  else
    printf '%s\n' "$wrapper" > "$MCC_ROOT/mcc" && chmod +x "$MCC_ROOT/mcc"
    echo "/usr/local/bin에 쓸 권한이 없어 $MCC_ROOT/mcc 로 만들었다(sudo로 다시 실행하면 'mcc'로 부를 수 있다)."
  fi
fi

echo
echo "완료. 조각 파일(*.part-*)은 지워도 된다."
echo "남은 일: $APP_DIR/configs/default.yaml 의 translate.api 에 LiteLLM 주소와 모델 이름을 적고 컬럼명을 맞춘다."
echo "실행 예) mcc --help"
echo "        LITELLM_API_KEY=sk-... mcc translate --input /data/train/fa50_ko.xlsx --output /data/train/fa50_aug.xlsx"
echo "        mcc train"
