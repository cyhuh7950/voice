# melotts TTS (MeloTTS, PyTorch CPU) — linux/arm64
#
# 빌드 컨텍스트는 상위 폴더다: docker build -f melotts/Dockerfile ..
#
# 의존성이 4개 엔진 중 가장 까다롭다. 정리하면:
#  - torch: aarch64 의 PyPI 기본 휠은 CUDA 빌드라 5GB 넘게 끌고 온다 → +cpu 휠 고정
#  - torchaudio: aarch64 에 +cpu 로컬버전이 없어 일반 휠을 --no-deps 로 넣는다
#  - MeloTTS 본체: setup.py 가 transformers==4.27.4 등 오래된 핀을 박아둬서 --no-deps
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    NLTK_DATA=/opt/nltk_data \
    PORT=8202

# git: MeloTTS 를 저장소에서 직접 설치 (PyPI 배포본이 없다)
# libsndfile1: soundfile / librosa 네이티브 의존성
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl git libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY melotts/requirements.txt ./requirements.txt
RUN pip install --upgrade pip wheel setuptools \
 && pip install -r requirements.txt

# 위 주석의 이유로 이 둘만 --no-deps 로 따로 넣는다.
RUN pip install --no-deps "torchaudio==2.9.1" \
 && pip install --no-deps "git+https://github.com/myshell-ai/MeloTTS.git"

# g2p_en 이 쓰는 NLTK 데이터를 이미지에 미리 넣어 런타임 다운로드를 없앤다.
# /models 는 볼륨이라 빌드 시 넣어도 마운트에 가려지므로 /opt 에 둔다.
RUN python -m nltk.downloader -d /opt/nltk_data \
      averaged_perceptron_tagger averaged_perceptron_tagger_eng cmudict punkt punkt_tab

COPY _common/voiceapi.py ./voiceapi.py
COPY melotts/server.py ./server.py

RUN mkdir -p /models
VOLUME ["/models"]

EXPOSE 8202
# 첫 기동 때 체크포인트와 BERT 를 받아야 해서 다른 엔진보다 더 여유를 준다.
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=5 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["python", "-u", "server.py"]
