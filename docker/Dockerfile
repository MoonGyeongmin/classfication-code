# 인터넷이 되는 PC에서 빌드한 뒤 docker save로 반입한다(docker/build_offline.sh).
# 모델 가중치는 이미지에 넣지 않고 /models로 마운트한다(이미지 크기와 교체 편의 때문).
# Docker Hub 요청 제한에 걸리면 BASE_IMAGE=mirror.gcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 로 바꾼다.
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    TZ=Asia/Seoul

RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip tzdata \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock .
# 사내 프록시가 TLS를 가로채는 PC라면 CA_CERT=/경로/ca.crt 로 인증서를 넘긴다(이미지에는 남지 않음).
RUN --mount=type=secret,id=ca_cert \
    if [ -s /run/secrets/ca_cert ]; then export PIP_CERT=/run/secrets/ca_cert; fi \
    && python -m pip install --upgrade pip && python -m pip install -r requirements.lock

COPY mcc ./mcc
COPY codebook ./codebook
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

ENTRYPOINT ["python", "-m", "mcc"]
CMD ["--help"]
