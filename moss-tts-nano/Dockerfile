# moss-tts-nano tts — linux/x86_64, CPU 전용 (ONNX Runtime)
#
# 빌드 컨텍스트는 상위 폴더다 (_common/voiceapi.py 를 함께 넣기 때문):
#   docker build -f moss-tts-nano/Dockerfile ..
#
# GPU 를 쓰지 않지만 베이스는 다른 엔진과 같은 것을 쓴다. ffmpeg / fastapi / uvicorn /
# torch(참조 음성 처리에 쓰인다) 가 이미 들어 있고, 레이어를 공유해 디스크를 아낀다.
#   docker build -f _base/Dockerfile -t voice-rocm-base:latest _base
FROM voice-rocm-base:latest

ENV PORT=8207

WORKDIR /app
COPY moss-tts-nano/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# 업스트림이 torch 를 재설치해 ROCm 빌드를 덮어쓰지 않았는지 빌드 시점에 확인한다.
RUN python3 -c "import torch, sys; \
    v = torch.__version__; \
    sys.exit(0) if 'rocm' in v else sys.exit('torch 가 ROCm 빌드가 아니다: ' + v)"

COPY _common/voiceapi.py ./voiceapi.py
COPY moss-tts-nano/server.py ./server.py
# 참조 음성. 내장 프리셋에 한국어 음색이 없어 한국어는 이쪽이 품질이 낫다.
COPY moss-tts-nano/voices/ ./voices/
# 업스트림이 라이브러리 API 를 문서화하지 않아 저장소를 통째로 쓴다.
# onnx_tts_runtime.py 는 패키지 안이 아니라 저장소 최상위에 있는 모듈이다.
COPY moss-tts-nano/repo/ ./repo/

RUN mkdir -p /models
# onnx_tts_runtime 은 자기 저장소 밑의 models/ 를 기본 경로로 보고, 그 경우에만 ONNX 자산을
# 자동으로 받는다. 그 자리를 /models 볼륨으로 연결해 받은 모델이 컨테이너 밖에 남게 한다.
RUN rm -rf /app/repo/models && ln -s /models /app/repo/models
VOLUME ["/models"]

EXPOSE 8207
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=5 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["python3", "-u", "server.py"]
