# qwen3-tts tts — linux/x86_64, AMD ROCm (WSL/ROCDXG, RX 7800 XT / gfx1101)
#
# 빌드 컨텍스트는 상위 폴더다 (_common/voiceapi.py 를 함께 넣기 때문):
#   docker build -f qwen3-tts/Dockerfile ..
#
# 베이스에 torch(ROCm) · ffmpeg · ROCm 런타임 의존 라이브러리 · fastapi/uvicorn 이 들어 있다:
#   docker build -f _base/Dockerfile -t voice-rocm-base:latest _base
FROM voice-rocm-base:latest

ENV PORT=8205

# qwen-tts 가 sox 를 의존한다 (오디오 후처리).
RUN apt-get update \
 && apt-get install -y --no-install-recommends sox libsox-fmt-all \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY qwen3-tts/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# requirements 설치 과정에서 의존성이 torch 를 PyPI 기본 휠(CUDA 빌드)로 갈아끼우면
# GPU 를 조용히 잃는다. 빌드 시점에 잡아서 터뜨린다.
RUN python3 -c "import torch, sys; \
    v = torch.__version__; \
    sys.exit(0) if 'rocm' in v else sys.exit('torch 가 ROCm 빌드가 아니다: ' + v)"

COPY _common/voiceapi.py ./voiceapi.py
COPY qwen3-tts/server.py ./server.py

RUN mkdir -p /models
VOLUME ["/models"]

EXPOSE 8205
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=5 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["python3", "-u", "server.py"]
