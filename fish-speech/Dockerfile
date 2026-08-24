# fish-speech tts — linux/x86_64, AMD ROCm (WSL/ROCDXG, RX 7800 XT / gfx1101)
#
# 빌드 컨텍스트는 상위 폴더다 (_common/voiceapi.py 를 함께 넣기 때문):
#   docker build -f fish-speech/Dockerfile ..
#
# 베이스에 torch(ROCm) · ffmpeg · ROCm 런타임 의존 라이브러리 · fastapi/uvicorn 이 들어 있다:
#   docker build -f _base/Dockerfile -t voice-rocm-base:latest _base
FROM voice-rocm-base:latest

ENV PORT=8206

# pyaudio 가 portaudio 헤더를 요구한다 (업스트림 의존성에 들어 있다).
RUN apt-get update \
 && apt-get install -y --no-install-recommends portaudio19-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY fish-speech/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# 업스트림 의존성 중 하나가 torch 를 PyPI 기본 휠(CUDA 빌드)로 갈아끼우면 GPU 를 잃는다.
# 특히 fish-speech 는 pyproject 에 torch==2.8.0 을 못박아두고 있어 위험이 크다.
# 빌드 시점에 잡아서 터뜨린다.
RUN python3 -c "import torch, sys; \
    v = torch.__version__; \
    sys.exit(0) if 'rocm' in v else sys.exit('torch 가 ROCm 빌드가 아니다: ' + v)"

COPY _common/voiceapi.py ./voiceapi.py
COPY fish-speech/server.py ./server.py
# 참조 음성. S2 는 프리셋 음색이 없어 이게 없으면 합성 자체가 안 된다.
COPY fish-speech/voices/ ./voices/
# 업스트림이 라이브러리 API 를 문서화하지 않아 저장소를 통째로 쓴다.
# 코드가 `from tools.server...` 처럼 저장소 루트 기준으로 import 하고,
# pyrootutils 가 .project-root 파일을 찾으므로 그것까지 함께 들어가야 한다.
COPY fish-speech/repo/ ./repo/

RUN mkdir -p /models
VOLUME ["/models"]

EXPOSE 8206
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=5 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["python3", "-u", "server.py"]
