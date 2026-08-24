# zipformer stt — linux/x86_64, CPU 전용
#
# 빌드 컨텍스트는 상위 폴더다 (_common/voiceapi.py 를 함께 넣기 때문):
#   docker build -f zipformer/Dockerfile ..
#
# GPU 를 쓰지 않지만 베이스는 다른 엔진과 같은 것을 쓴다. ffmpeg / fastapi / uvicorn 이
# 이미 들어 있고, 이미지 레이어를 공유해 디스크를 아낀다.
#   docker build -f _base/Dockerfile -t voice-rocm-base:latest _base
FROM voice-rocm-base:latest

ENV PORT=8105

WORKDIR /app
COPY zipformer/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY _common/voiceapi.py ./voiceapi.py
COPY zipformer/server.py ./server.py

RUN mkdir -p /models
VOLUME ["/models"]

EXPOSE 8105
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=5 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["python3", "-u", "server.py"]
