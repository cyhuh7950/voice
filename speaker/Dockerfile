# speaker 화자 임베딩 (ONNX Runtime + kaldi-native-fbank) — linux/arm64 CPU
#
# 빌드 컨텍스트는 상위 폴더다 (_common/voiceapi.py 를 함께 넣기 때문):
#   docker build -f speaker/Dockerfile ..
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SPEAKER_MODEL_DIR=/models \
    PORT=8301

# ffmpeg: 업로드된 webm/mp3/m4a 를 16kHz mono PCM 으로 디코딩
# curl: 헬스체크 + 모델 다운로드 재시도 경로
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY speaker/requirements.txt ./requirements.txt
RUN pip install --upgrade pip wheel setuptools \
 && pip install -r requirements.txt

COPY _common/voiceapi.py ./voiceapi.py
COPY speaker/server.py ./server.py

RUN mkdir -p /models
VOLUME ["/models"]

EXPOSE 8301
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=5 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["python", "-u", "server.py"]
