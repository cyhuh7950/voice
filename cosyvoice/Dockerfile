# cosyvoice tts — linux/x86_64, AMD ROCm (WSL/ROCDXG, RX 7800 XT / gfx1101)
#
# 빌드 컨텍스트는 cosyvoice/ 폴더다 (repo/ 서브모듈을 통째로 넣기 때문):
#   docker build -f Dockerfile .
#
# 베이스에 torch(ROCm) · ffmpeg · ROCm 런타임 의존 라이브러리 · fastapi/uvicorn 이 들어 있다:
#   docker build -f _base/Dockerfile -t voice-rocm-base:latest _base
FROM voice-rocm-base:latest

ENV PORT=8203

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# matcha-tts 는 빌드 격리(PEP 517) 안에서 numpy 를 새로 소스 빌드하려 든다. Python 3.12 에는
# 그 조합의 numpy 휠이 없어 실패하므로, 격리를 끄고 이미 깔린 numpy 1.26.4 를 그대로 쓰게 한다.
RUN pip install --no-build-isolation --no-deps matcha-tts==0.0.7.2

# requirements 설치 과정에서 의존성이 torch 를 PyPI 기본 휠(CUDA 빌드)로 갈아끼우면
# GPU 를 조용히 잃는다. 빌드 시점에 잡아서 터뜨린다.
RUN python3 -c "import torch, sys; \
    v = torch.__version__; \
    sys.exit(0) if 'rocm' in v else sys.exit('torch 가 ROCm 빌드가 아니다: ' + v)"

COPY voiceapi.py ./voiceapi.py
COPY server.py ./server.py
COPY repo/ ./repo/
# CosyVoice3 는 SFT 프리셋 보이스가 없고 zero-shot 복제만 되므로 참조 음성이 필수다.
# server.py 의 VOICES_DIR 이 /app/voices 를 본다. (CUDA 판에서는 이 COPY 가 빠져 있어
#  컨테이너 안에서 "알 수 없는 voice: default" 로 합성이 전부 실패한다)
COPY voices/ ./voices/

RUN mkdir -p /models
VOLUME ["/models"]

# 로딩 실측: ROCm/WSL 에서 83초 (CUDA 판 기록은 1014초였는데, 그건 9GB 체크포인트를
# 콜드 디스크에서 읽은 값으로 보인다). 콜드 캐시를 감안해 600초로 잡는다.
EXPOSE 8203
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=5 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["python3", "-u", "server.py"]
