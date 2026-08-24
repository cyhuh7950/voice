# __ENGINE__ __KIND__ — linux/x86_64, AMD ROCm (WSL/ROCDXG)
#
# 빌드 컨텍스트는 상위 폴더다 (_common/voiceapi.py 를 함께 넣기 때문):
#   docker build -f __ENGINE__/Dockerfile ..
#
# 베이스는 _base/Dockerfile 로 만든 공용 이미지다. torch(ROCm) 6GB 레이어와
# ffmpeg / ROCm 런타임 의존 라이브러리 / fastapi·uvicorn 이 이미 들어 있다.
#   docker build -f _base/Dockerfile -t voice-rocm-base:latest _base
FROM voice-rocm-base:latest

ENV PORT=__PORT__

WORKDIR /app
COPY __ENGINE__/requirements.txt ./requirements.txt
# TODO: 엔진 고유 의존성만 requirements.txt 에 남긴다.
#   torch/torchaudio/fastapi/uvicorn/numpy/soundfile 는 베이스에 있으므로 다시 적지 말 것.
#   특히 torch 를 다시 적으면 PyPI 의 CUDA 빌드가 ROCm 빌드를 덮어써 GPU 를 잃는다.
RUN pip install -r requirements.txt

COPY _common/voiceapi.py ./voiceapi.py
COPY __ENGINE__/server.py ./server.py

RUN mkdir -p /models
VOLUME ["/models"]

EXPOSE __PORT__
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=5 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["python", "-u", "server.py"]
