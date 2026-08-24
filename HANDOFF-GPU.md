# GPU 서버(WSL) 확장 인수 문서 — AMD ROCm 판

이 문서는 `ysna-server`(오라클 ARM64, CPU 전용)에 구축한 STT/TTS 컨테이너 구조를
**WSL GPU 서버(`daon@DESKTOP-6SUOAH9`)** 로 확장하기 위한 인수 자료다.

용도: **내부 개발 전용.** 외부 노출·상업 배포를 전제하지 않는다.
따라서 비상업 라이선스 엔진(XTTS-v2, Fish Speech)도 포함한다.

> **개정 이력**
> 이 문서의 이전 판은 NVIDIA RTX 5070(CUDA 12.8) 을 전제로 작성됐다.
> GPU 가 **AMD Radeon RX 7800 XT** 로 교체되면서 전면 개정했다. CUDA 전용 내용은
> 모두 ROCm/HIP 로 대체됐고, 아래 "2. ROCm 환경" 절의 수치는 이 머신에서 직접 측정한 값이다.

---

## 1. 그대로 재사용하는 것

새로 설계할 필요 없다. 아래는 GPU 종류와 무관하게 동작한다.

| 파일 | 역할 |
|---|---|
| `_common/voiceapi.py` | HTTP 규격 전체. OpenAI Audio API 호환 엔드포인트, ffmpeg 오디오 변환, API 키 인증, `/health`·`/info`, 모델 로딩 상태 관리 |
| `voicectl.sh` | 폴더 자동 탐색 기반 제어 (list/start/stop/status/logs/test/new) |
| `_template/` | 새 엔진 스캐폴딩. `voicectl.sh new <이름> <stt\|tts> <포트>` |
| `_common/mkvenv.sh` | 폴더별 venv 생성 |

**새로 추가된 것**

| 파일 | 역할 |
|---|---|
| `_base/Dockerfile` | 모든 GPU 엔진이 공유하는 ROCm 런타임 베이스 이미지 (`voice-rocm-base:latest`) |

**규약**
- 포트: STT `81xx`, TTS `82xx`
- 엔진 1개 = 폴더 1개 = 컨테이너 1개. 엔진 간 `depends_on` 없음
- 설정 단일 소스: `<엔진>/engine.env` (voicectl 이 `--env-file` 로 넘긴다)
- 모델 캐시: `<엔진>/models/` 볼륨 마운트
- `restart: unless-stopped` (`always` 금지 — 끈 것은 꺼져 있어야 함)
- **필요한 엔진만 기동.** 전체 기동 명령을 만들지 않는다
- 컨테이너는 기존 `proxy-network` 에 external 로 붙인다

**엔진이 구현할 것은 2개뿐**

```python
# STT: 16kHz mono float32 numpy → dict
def transcribe(audio, *, language, prompt, temperature, task) -> dict:
    return {"text": "...", "language": "ko", "segments": []}

# TTS: 텍스트 → (mono float32 numpy, sample_rate)
def synthesize(text, *, voice, language, speed) -> tuple[np.ndarray, int]:
    return wav, 24000
```

---

## 2. ROCm 환경 (구축 완료 · 실측)

### 2.1 하드웨어 / 호스트

| 항목 | 값 |
|---|---|
| GPU | AMD Radeon RX 7800 XT — **gfx1101**, 60 CU, **VRAM 16GB** |
| CPU / RAM | AMD Ryzen 5 9600X (6C/12T) / 23GB + swap 64GB |
| Windows | 11 Pro build 26200.8875 |
| AMD 드라이버 | **Adrenalin 26.7.1** (ROCDXG 요구치 26.2.2 이상 충족 — 별도 설치 불필요했음) |
| WSL | 2.7.3.0 / 커널 6.6.114.1 |
| 배포판 | Ubuntu 24.04.3 LTS (x86_64) |

### 2.2 WSL 의 ROCm 은 커널 드라이버를 쓰지 않는다

이게 네이티브 리눅스와 가장 크게 다른 점이고, 대부분의 삽질이 여기서 나온다.

```
네이티브 Linux :  앱 → ROCm 런타임 → libhsakmt → /dev/kfd, /dev/dri  (amdgpu 커널 모듈)
WSL            :  앱 → ROCm 런타임 → librocdxg → /dev/dxg → Windows 디스플레이 드라이버
```

따라서:
- `amdgpu-dkms` 를 **설치하면 안 된다.** WSL 에는 커널 모듈이 올라가지 않는다
- `/dev/kfd`, `/dev/dri` 는 존재하지 않는다. 컨테이너에 넘길 디바이스는 **`/dev/dxg` 하나**다
- `rocm-smi` 는 동작하지 않는다 (`amdgpu not found in modules`). **`amd-smi` 를 쓴다**
- ROCm 프로파일러/디버거는 지원되지 않는다 (아래 2.4 참고)

### 2.3 호스트 설치 절차 (재현용)

```bash
# 1) ROCm 리포 등록
wget https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/amdgpu-install_7.2.4.70204-1_all.deb
sudo apt install ./amdgpu-install_7.2.4.70204-1_all.deb
sudo apt update

# 2) ROCm 설치 — 커널 드라이버 제외가 핵심이다
sudo amdgpu-install --usecase=rocm --no-dkms -y      # 약 22GB, 10분대

# 3) ROCDXG (WSL 시임). 소스 빌드 불필요 — 프리빌트 deb 를 쓴다
#    https://github.com/ROCm/librocdxg/releases
sudo dpkg -i rocdxg-roct_1.2.1_amd64.deb             # librocdxg.so + dids.conf
sudo dpkg -i rocdxg-amd-smi-lib_1.2.1_amd64.deb      # /opt/rocm-wsl/bin/amd-smi
```

> `rocdxg-roct` **1.2.0 은 `dids.conf` 를 넣지 않는다.** 컨테이너 마운트에 필요하므로
> 1.2.1 이상을 쓸 것. librocdxg 공식 호환표 기준 RX 7800 XT 는 rocdxg 1.2.0 / ROCm 7.2.x
> 행에 명시돼 있고, 상위 버전은 누적 지원이다.

### 2.4 필수 환경변수 2개

`/etc/profile.d/rocm-wsl.sh` 에 넣어뒀다. **둘 다 없으면 GPU 를 못 쓴다.**

```bash
export HSA_ENABLE_DXG_DETECTION=1     # ROCm 7.13 미만에서 ROCDXG 감지를 켠다. 없으면 hsa_init 실패
export ROCPROFILER_REGISTER_ENABLED=0 # 없으면 torch.cuda.device_count() 에서 프로세스가 abort 한다
export PATH=/opt/rocm/bin:/opt/rocm-wsl/bin:$PATH
```

`ROCPROFILER_REGISTER_ENABLED=0` 을 빠뜨리면 이렇게 죽는다. 원인 파악이 어려우니 기억해둘 것:

```
F agent.cpp:1093] Found 0 rocprofiler agents and 2 HSA agents.
  HSA agents contained 2 internal node ids not found by rocprofiler: 0, 1
Aborted (core dumped)
```

### 2.5 검증 결과

```bash
rocminfo | grep -A3 'Agent 2'
#   Name: gfx1101 / Marketing Name: AMD Radeon RX 7800 XT / Device Type: GPU / Compute Unit: 60

/opt/rocm-wsl/bin/amd-smi monitor
#   GPU  POWER  GPU_T  GFX_CLK  GFX%  VRAM_USAGE
#     0   83 W  79 °C  1474MHz   16%   2.7/16.0 GB
```

| 측정 | 호스트 venv | 컨테이너 |
|---|---|---|
| `torch.cuda.is_available()` | True | True |
| matmul 4096² fp32 | 8.1 TFLOPS | — |
| matmul 4096² **fp16** | **47.4 TFLOPS** | **50.1 TFLOPS** |
| matmul 4096² bf16 | 44.6 TFLOPS | — |

컨테이너가 호스트보다 느리지 않다. ROCDXG 경유로 인한 성능 손실은 관측되지 않았다.

### 2.6 PyTorch

```bash
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch==2.13.0+rocm7.2 torchaudio
```

- 휠 자체가 **6.2GB** 다. 엔진마다 받으면 디스크가 남아나지 않아 공용 베이스 이미지로 분리했다
- ROCm 환경에서도 API 는 `torch.cuda.*` 그대로다 (HIP 백엔드로 매핑). `torch.version.hip` 로 구분한다
- **`requirements.txt` 에 `torch` 를 다시 적으면 안 된다.** PyPI 기본 휠(CUDA 빌드)이
  ROCm 빌드를 덮어써서 GPU 를 조용히 잃는다. 이건 이전 CUDA 판에서도 같은 함정이었다

### 2.7 ONNX Runtime — WSL 에서 GPU 가속 불가

AMD 리포(`repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/`)가 내놓는 ONNX 휠은
`onnxruntime_migraphx` 뿐인데, AMD 공식 문서가 **"MIGraphX and mGPU configuration are not
currently supported by WSL"** 이라고 명시한다. 영향받는 엔진:

- **CosyVoice3** — speech tokenizer / flow decoder 의 ONNX 부분은 CPU 폴백.
  LLM·flow·hift 는 PyTorch 라 GPU 를 쓴다
- **Zipformer(sherpa-onnx)** — 전면 CPU. 원래 온디바이스용 모델이라 CPU 로도 실용적이다

PyPI 의 `onnxruntime-rocm`(MIGraphX 가 아닌 ROCm EP) 로 우회 가능한지는 별도 검증 대상.

---

## 3. 컨테이너에서 GPU 쓰기

NVIDIA 판의 `deploy.resources.reservations.devices: [driver: nvidia]` 블록은 **전부 삭제한다.**
`nvidia-container-toolkit` 도 이 환경에서는 쓰이지 않는다
(GPU 교체 직후 기존 컨테이너가 `nvidia-container-cli: WSL environment detected but no adapters
were found` 로 exit 128 을 낸 것이 이 때문이다).

대신 `docker-compose.yml` 에 이 블록을 넣는다. `_template/docker-compose.yml` 에 반영해뒀다.

```yaml
    volumes:
      - ./models:/models
      - /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so:ro
      - /opt/rocm/lib/librocdxg.so:/usr/lib/librocdxg.so:ro
      - /opt/rocm/share/rocdxg/dids.conf:/usr/share/rocdxg/dids.conf:ro
    devices:
      - /dev/dxg
    cap_add:
      - SYS_PTRACE
    security_opt:
      - seccomp:unconfined
    ipc: host
    shm_size: ${SHM_SIZE:-8gb}
```

**이미지 안에 ROCm 을 설치할 필요는 없다.** torch 휠이 자체 ROCm 런타임을 번들하고 있어서,
위 3개 라이브러리만 호스트에서 넣어주면 컨테이너 안에서 GPU 가 그대로 잡힌다 (2.5 의 실측이 그것이다).
22GB `/opt/rocm` 을 이미지에 넣거나 마운트할 이유가 없다.

다만 `ubuntu:24.04` 최소 이미지에는 ROCm 런타임이 `dlopen` 하는 것들이 빠져 있어
**`libatomic1 libnuma1 libdrm2 libdrm-amdgpu1 libelf1`** 을 apt 로 넣어야 한다.
없으면 `import torch` 가 `libatomic.so.1: cannot open shared object file` 로 깨진다.

### 공용 베이스 이미지

```bash
docker build -f _base/Dockerfile -t voice-rocm-base:latest _base
```

여기에 torch(ROCm) · ffmpeg · ROCm 런타임 의존 라이브러리 · fastapi/uvicorn 이 들어 있다.
각 엔진 Dockerfile 은 `FROM voice-rocm-base:latest` 로 시작하고 **자기 고유 의존성만** 설치한다.

---

## 4. 설치 대상 엔진 8종

포트는 확정값이다.

### STT (81xx)

**① Whisper large-v3 — 포트 8103**
- **주의: faster-whisper 는 ROCm 에서 GPU 를 못 쓴다.** 백엔드인 CTranslate2 가 CUDA/CPU 만
  지원하고 ROCm 백엔드가 없다. `ctranslate2-rocm` 은 PyPI 에 없고 커뮤니티 소스 빌드뿐이다
- 따라서 ROCm 에서는 백엔드를 바꿔야 한다. 후보:
  - `transformers` 의 `WhisperForConditionalGeneration` + torch-ROCm (가장 안전. 공식 지원 경로)
  - `openai-whisper` (원본) + torch-ROCm
  - `whisper.cpp` HIP/Vulkan 빌드
- 모델 `large-v3`, fp16
- **기존 `whisper-gpu/models/` 3.0GB 캐시는 재사용할 수 없다.** 받아둔 것은
  `Systran/faster-whisper-large-v3` 즉 CTranslate2 포맷이라 transformers 가 읽지 못한다.
  `openai/whisper-large-v3`(safetensors)를 새로 받아야 한다
- 라이선스: 코드 MIT, 모델 MIT

**② Whisper streaming — 포트 8104**
- [WhisperLive](https://github.com/collabora/WhisperLive) — WebSocket, 지연 약 500~800ms
- [whisper_streaming](https://github.com/ufal/whisper_streaming) — LocalAgreement 정책, 자체 적응형 지연
- 둘 다 faster-whisper 백엔드를 전제하므로 ①과 같은 ROCm 제약을 받는다
- **주의**: 스트리밍은 WebSocket 이라 `voiceapi.py` 의 HTTP 계약으로 안 덮인다.
  `create_app()` 에 WS 엔드포인트를 추가하거나 이 엔진만 별도 규격으로 갈지 결정 필요
- 음성 대화형 번역이 목표라면 이 엔진이 핵심이다 (배치 STT 는 발화 끝까지 기다려야 함)

**⑦ Zipformer (sherpa-onnx) — 포트 8105**
- 한국어 Zipformer. `pip install sherpa-onnx`
- ONNX Runtime GPU 가 WSL 에서 막혀 있으므로 (2.7) **CPU 모드로 간다.**
  원래 모바일/온디바이스용 모델이라 CPU 로도 실시간을 넘는다

### TTS (82xx)

**③ CosyVoice3 — 포트 8203**
- 모델 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`. **이미 받아둔 `cosyvoice/models/` 9.1GB 를 재사용한다**
- 기존 폴더가 CUDA(cu128) 로 구성돼 있으므로 torch 를 ROCm 휠로 교체하고
  `onnxruntime-gpu` 를 제거한다 (CPU onnxruntime 폴백)
- 양방향 스트리밍 TTS, 3초 오디오로 zero-shot 음성 복제
- 서브모듈(`--recursive`) 누락이 흔한 실패 원인

**④ XTTS-v2 — 포트 8204**
- **원본 `coqui-ai/TTS` 대신 유지보수 포크를 쓸 것**: `pip install coqui-tts` ([idiap/coqui-ai-TTS](https://github.com/idiap/coqui-ai-TTS))
  import 경로는 동일 (`from TTS.api import TTS`)
- 17개 언어, 음성 복제
- 라이선스: 코드 MPL 2.0 / **가중치 CPML 비상업**. Coqui Inc. 가 2024-01 폐업해
  **상업 라이선스를 살 방법이 없다.** 상업화 시 교체 대상

**⑤ Qwen3-TTS 0.6B — 포트 8205**
- `pip install -U qwen-tts`. 모델 2종:
  - `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (2.52GB) — 오디오 입력으로 빠른 음성 복제
  - `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` — 프리셋 음색 9종, 자연어 지시로 스타일 제어, 10개 언어
- 순수 PyTorch 라 ROCm 이식 위험이 가장 낮다

**⑥ Fish Speech — 포트 8206**
- [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech). 최신은 **S2**
- 라이선스: **FISH AUDIO RESEARCH LICENSE** — 연구·비상업 무료, 상업은 별도 유료 라이선스
  (XTTS-v2 와 달리 구매 경로는 존재한다)

**⑧ MOSS-TTS-Nano — 포트 8207**
- [OpenMOSS/MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) 100M. 경량 모델
- 순수 PyTorch. ROCm 전용 venv 에서 구동

---

## 5. VRAM — 한 번에 GPU 엔진 하나만 띄운다

**운용 방침(확정): GPU 엔진을 여러 개 동시에 올리지 않는다.**
따라서 16GB 를 한 엔진이 온전히 쓰는 것을 전제로 설계하면 된다.
동시 상주를 맞추려고 양자화하거나 모델을 줄일 이유가 없다.

이 방침은 지켜야 하는 것이지 권고가 아니다. 실제로 whisper-gpu·cosyvoice·xtts 를 켜둔 채
fish-speech(3B)를 올리려다 VRAM 10/16GB 에서 막혔다.

- CPU 전용 엔진(zipformer 8105, moss-tts-nano 8207)은 **VRAM 을 쓰지 않으므로 이 제약 밖이다.**
  GPU 엔진과 무관하게 상시 띄워둘 수 있다
- 성능 측정은 **반드시 단독 기동 상태에서** 할 것. 여러 엔진이 떠 있으면 GPU 경합으로
  수치가 흐려진다 (이번 구축에서 실제로 겪었다)
- `engine.env` 의 `PRELOAD=0` 은 첫 요청까지 모델을 올리지 않는다. 엔진을 자주 바꿔가며
  쓸 때 유용하다

RTF 는 **단독 기동 상태**에서 워밍업(첫 요청) 이후를 잰 값이다. 1보다 작으면 실시간보다 빠르다.

| 엔진 | 포트 | 종류 | 모델 | 장치 | VRAM | RTF | 상태 |
|---|---|---|---|---|---|---|---|
| whisper-gpu | 8103 | STT | large-v3 | GPU | 4.2GB | **0.338** | 검증 완료 |
| whisper-streaming | 8104 | STT | large-v3 | GPU | - | - | 구현 완료, 검증 대기 |
| zipformer | 8105 | STT | ko zipformer(int8) | **CPU** | 0 | **0.010** | 검증 완료 |
| cosyvoice | 8203 | TTS | Fun-CosyVoice3-0.5B | GPU | 4.2GB | **1.700** | 검증 완료 |
| xtts | 8204 | TTS | XTTS-v2 | GPU | 약 2GB | 3.362 | 검증 완료 |
| qwen3-tts | 8205 | TTS | Qwen3-TTS-0.6B | GPU | 3.1GB | 6.500 | 검증 완료 |
| fish-speech | 8206 | TTS | Fish Speech S2 (s2-pro) | GPU | **GPU 단독 필요** | - | 검증 대기 |
| moss-tts-nano | 8207 | TTS | MOSS-TTS-Nano-100M | **CPU** | 0 | 0.541 | 검증 완료 |

**엔진 선택 가이드 (실측 기준)**

- **실시간 대화형이면 zipformer(8105) + cosyvoice(8203)** 조합이다.
  zipformer 는 CPU 만 쓰면서 RTF 0.010 이라 GPU 를 TTS 에 온전히 넘길 수 있다
- **인식 정확도가 중요하면 whisper-gpu(8103).** zipformer 는 34배 빠르지만 오인식이 있다
  (실측: "합성"→"학생"). 띄어쓰기는 server.py 에서 토큰을 이어 붙여 해결했다
- **qwen3-tts 는 배치 합성용.** RTF 6.5 는 12Hz 프레임마다 다중 코드북을 순차 생성하는
  구조 탓이라 파인튜닝으로 개선되지 않는다. 실시간에는 cosyvoice 가 3.8배 빠르다
- **fish-speech 는 VRAM 요구가 가장 크다.** 3B + max_seq_len 32768 이라 KV 캐시만으로도
  수 GB 를 쓴다. 다른 GPU 엔진을 모두 내리고 띄워야 한다

**참조 음성(voice cloning)을 쓸 때 주의**

현재 `voices/default.wav` 는 4.17초인데 **XTTS-v2 는 이 길이에서 생성이 붕괴한다**
(실측: "안녕하세요"가 200회 넘게 반복). 6초 이상, 되도록 10초 이상의 깨끗한 음성을 쓸 것.
cosyvoice·moss-tts-nano 는 4.17초로도 정상 동작했지만 여유를 두는 편이 안전하다.

VRAM 확인:

```bash
/opt/rocm-wsl/bin/amd-smi monitor
```

---

## 6. 작업 순서

각 엔진마다 컨테이너 이전에 `_common/mkvenv.sh <엔진>` 으로 호스트 venv 에서 먼저 검증하면
의존성 문제를 훨씬 빨리 잡는다. ROCm 은 CUDA 보다 휠 조합이 예민해서 이 방식이 특히 유효하다.

1. ~~ROCm 스택 구축~~ **완료** (2절)
2. ~~공용 베이스 이미지~~ **완료** (`_base/Dockerfile`)
3. **CosyVoice3(8203)** — 모델이 이미 있어 이식만 하면 되고, 구조 검증이 가장 빠르다
4. **Qwen3-TTS(8205)**, **MOSS-TTS-Nano(8207)** — 순수 PyTorch 라 ROCm 위험이 낮다
5. **Whisper(whisper-gpu, 8103)** — 백엔드 교체 결정이 필요해 시간이 걸린다
6. **XTTS-v2(8204)**, **Fish Speech(8206)**
7. **Zipformer(8105)** — CPU 라 GPU 이슈와 무관
8. **Whisper streaming(8104)** — WebSocket 규격 결정이 필요하므로 마지막에

---

## 7. CPU 서버(ysna-server) 현황 참고

같은 구조로 4개가 돌고 있다. 실측치는 GPU 결과와 비교할 기준선이 된다.

| 엔진 | 포트 | 종류 | 메모리 | RTF | 비고 |
|---|---|---|---|---|---|
| whisper (base/int8) | 8101 | STT | 335MB | 1.12 | 한국어 정확 |
| moonshine (tiny-ko) | 8102 | STT | 187MB | 0.05~0.18 | 매우 빠름, **한국어 문장 끝 누락** |
| supertonic-3 | 8201 | TTS | 497MB | 1.93 | 31개 언어, 보이스 10종 |
| MeloTTS-KR | 8202 | TTS | 1.72GB | 4.8 | 느림 |

공개 엔드포인트(NPM 경유, Let's Encrypt):
`stt.whisper.sinsan.kr` / `stt.moonshine.sinsan.kr` / `tts.supertonic.sinsan.kr` / `tts.melotts.sinsan.kr`

전체 사용법과 API 규격은 같은 폴더의 `README.md` 참고.
