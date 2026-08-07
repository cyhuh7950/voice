# __ENGINE__ __KIND__ — linux/arm64 CPU
#
# 빌드 컨텍스트는 상위 폴더다 (_common/voiceapi.py 를 함께 넣기 때문):
#   docker build -f __ENGINE__/Dockerfile ..
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    PORT=__PORT__

# ffmpeg 는 오디오 디코딩/인코딩에 필수다. curl 은 헬스체크용.
# TODO: 엔진이 요구하는 네이티브 라이브러리를 추가한다.
#   soundfile/librosa → libsndfile1 / sounddevice → libportaudio2 / git 설치 → git
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

# 호스트의 폴더별 venv 와 같은 격리를 이미지 안에서도 유지한다.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY __ENGINE__/requirements.txt ./requirements.txt
RUN pip install --upgrade pip wheel setuptools \
 && pip install -r requirements.txt

COPY _common/voiceapi.py ./voiceapi.py
COPY __ENGINE__/server.py ./server.py

RUN mkdir -p /models
VOLUME ["/models"]

EXPOSE __PORT__
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=5 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["python", "-u", "server.py"]
