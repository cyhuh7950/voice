# supertonic TTS (ONNX Runtime) — linux/arm64 CPU
#
# 빌드 컨텍스트는 상위 폴더다: docker build -f supertonic/Dockerfile ..
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SUPERTONIC_MODEL_DIR=/models \
    HF_HOME=/models/hf \
    PORT=8201

# ffmpeg: mp3/opus/flac 출력 인코딩
# libsndfile1: supertonic 이 쓰는 soundfile 의 네이티브 의존성
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY supertonic/requirements.txt ./requirements.txt
RUN pip install --upgrade pip wheel setuptools \
 && pip install -r requirements.txt

COPY _common/voiceapi.py ./voiceapi.py
COPY supertonic/server.py ./server.py

RUN mkdir -p /models
VOLUME ["/models"]

EXPOSE 8201
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=5 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["python", "-u", "server.py"]
