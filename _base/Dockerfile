# voice-rocm-base — 모든 GPU 엔진이 공유하는 ROCm 런타임 베이스.
#
#   docker build -f _base/Dockerfile -t voice-rocm-base:latest _base
#
# WSL 의 ROCm 은 커널 드라이버(amdgpu/KFD)가 아니라 ROCDXG(librocdxg) 시임을 거쳐 /dev/dxg 로
# 붙는다. 그래서 이 이미지 안에는 ROCm 을 설치하지 않는다 — torch 휠이 자체 ROCm 런타임을
# 번들하고 있고, 호스트에서 librocdxg.so / dids.conf / libdxcore.so 3개만 마운트하면 된다.
# (검증: 컨테이너 안 fp16 matmul 50.1 TFLOPS, 호스트와 동일)
#
# 즉 nvidia/cuda 베이스도, 22GB /opt/rocm 도 이미지에 들어가지 않는다.
FROM ubuntu:24.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    HSA_ENABLE_DXG_DETECTION=1 \
    ROCPROFILER_REGISTER_ENABLED=0

# ffmpeg: 오디오 디코딩/인코딩(voiceapi.py 필수). curl: 헬스체크.
# libatomic1/libnuma1/libdrm2/libdrm-amdgpu1/libelf1: ROCm 런타임이 dlopen 하는 것들.
#   (ubuntu:24.04 최소 이미지에는 없어서 torch import 자체가 libatomic.so.1 로 깨진다)
# libsndfile1: soundfile/librosa. git: 일부 엔진이 소스에서 모델/코드를 받는다.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev \
      ffmpeg curl git \
      libatomic1 libnuma1 libdrm2 libdrm-amdgpu1 libelf1 libsndfile1 \
      g++ \
 && rm -rf /var/lib/apt/lists/*

# 호스트의 폴더별 venv 와 같은 격리를 이미지 안에서도 유지한다.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ROCm 7.2 빌드 torch. 호스트에 설치된 ROCm 과 같은 계열이어야 한다(7.2.4).
# 이 레이어가 6GB 대라 엔진마다 다시 받지 않도록 베이스로 분리한 것이다.
RUN pip install --upgrade pip wheel setuptools \
 && pip install --index-url https://download.pytorch.org/whl/rocm7.2 \
      torch==2.13.0+rocm7.2 torchaudio

# 모든 엔진이 공유하는 HTTP 레이어 의존성.
RUN pip install \
      "fastapi>=0.115,<1" \
      "uvicorn[standard]>=0.30" \
      "python-multipart>=0.0.9" \
      "numpy>=1.24,<3" \
      soundfile

WORKDIR /app
RUN mkdir -p /models
