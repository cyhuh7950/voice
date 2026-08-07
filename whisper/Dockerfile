# whisper STT (faster-whisper / CTranslate2) — linux/arm64 CPU
#
# 빌드 컨텍스트는 상위 폴더다. _common/voiceapi.py 를 함께 넣기 때문.
#   docker build -f whisper/Dockerfile ..
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    PORT=8101

# ffmpeg: 업로드된 임의 포맷(webm/mp3/m4a)을 16kHz mono PCM 으로 디코딩
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

# 호스트의 폴더별 venv 와 같은 격리를 이미지 안에서도 유지한다.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY whisper/requirements.txt ./requirements.txt
RUN pip install --upgrade pip wheel setuptools \
 && pip install -r requirements.txt

COPY _common/voiceapi.py ./voiceapi.py
COPY whisper/server.py ./server.py

RUN mkdir -p /models
VOLUME ["/models"]

EXPOSE 8101
# 모델 다운로드가 걸릴 수 있어 start-period 를 넉넉히 준다.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=5 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["python", "-u", "server.py"]
