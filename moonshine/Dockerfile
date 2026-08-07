# moonshine STT (moonshine-voice, 내장 ONNX 런타임) — linux/arm64 CPU
#
# 빌드 컨텍스트는 상위 폴더다: docker build -f moonshine/Dockerfile ..
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MOONSHINE_VOICE_CACHE=/models \
    PORT=8102

# ffmpeg: 업로드 오디오 디코딩
# libportaudio2: moonshine-voice 가 끌고 오는 sounddevice 가 import 시점에 찾는다
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl libportaudio2 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY moonshine/requirements.txt ./requirements.txt
RUN pip install --upgrade pip wheel setuptools \
 && pip install -r requirements.txt

COPY _common/voiceapi.py ./voiceapi.py
COPY moonshine/server.py ./server.py

RUN mkdir -p /models
VOLUME ["/models"]

EXPOSE 8102
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=5 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["python", "-u", "server.py"]
