# GPU 서버(WSL) 확장 인수 문서

이 문서는 `ysna-server`(오라클 ARM64, CPU 전용)에 구축한 STT/TTS 컨테이너 구조를
**WSL GPU 서버(`daon@SINSAN`)** 로 확장하기 위한 인수 자료다.

용도: **내부 개발 전용.** 외부 노출·상업 배포를 전제하지 않는다.
따라서 비상업 라이선스 엔진(XTTS-v2, Fish Speech)도 포함한다.

---

## 1. 그대로 재사용하는 것

새로 설계할 필요 없다. 아래는 GPU 환경에서도 수정 없이 동작한다.

| 파일 | 역할 |
|---|---|
| `_common/voiceapi.py` | HTTP 규격 전체. OpenAI Audio API 호환 엔드포인트, ffmpeg 오디오 변환, API 키 인증, `/health`·`/info`, 모델 로딩 상태 관리 |
| `voicectl.sh` | 폴더 자동 탐색 기반 제어 (list/start/stop/status/logs/test/new) |
| `_template/` | 새 엔진 스캐폴딩. `voicectl.sh new <이름> <stt\|tts> <포트>` |
| `_common/mkvenv.sh` | 폴더별 venv 생성 |

**규약**
- 포트: STT `81xx`, TTS `82xx`
- 엔진 1개 = 폴더 1개 = 컨테이너 1개. 엔진 간 `depends_on` 없음
- 설정 단일 소스: `<엔진>/engine.env` (voicectl 이 `--env-file` 로 넘긴다)
- 모델 캐시: `<엔진>/models/` 볼륨 마운트
- `restart: unless-stopped` (`always` 금지 — 끈 것은 꺼져 있어야 함)
- **필요한 엔진만 기동.** 전체 기동 명령을 만들지 않는다
- 컨테이너는 기존 `proxy-network` 에 external 로 붙인다 (그 서버에 NPM 이 있다면)

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

## 2. 설치할 엔진 6종 (조사 완료)

포트는 제안값이다. 머신이 다르므로 8101/8201 을 재사용해도 무방하다.

### STT

**① Whisper large-v3 (faster-whisper) — 포트 8101**
- `pip install faster-whisper`
- 모델 `large-v3`, GPU 에서는 `compute_type="float16"` (CPU 의 `int8` 대신)
- CUDA 12 + **cuDNN 9** 필요. CTranslate2 4.x 가 cuDNN 9 를 링크한다
- ARM CPU 에서 base/int8 RTF 1.12 였다 → GPU large-v3 은 RTF 0.1 내외 기대
- 라이선스: 코드 MIT, 모델 MIT
- **기존 `whisper/server.py` 를 그대로 쓰고 `engine.env` 만 바꾸면 된다** (모델·compute_type)

**② Whisper streaming — 포트 8103**
- 구현체 2개 중 선택:
  - [WhisperLive](https://github.com/collabora/WhisperLive) — 서버/클라이언트 구조, **WebSocket**, faster-whisper·TensorRT·OpenVINO 백엔드, 지연 약 500~800ms
  - [whisper_streaming](https://github.com/ufal/whisper_streaming) — LocalAgreement 정책, 자체 적응형 지연, faster-whisper 백엔드 권장
- **주의**: 스트리밍은 WebSocket 이라 `voiceapi.py` 의 HTTP 계약으로 안 덮인다.
  `create_app()` 에 WS 엔드포인트를 추가하거나 이 엔진만 별도 규격으로 갈지 결정 필요
- 음성 대화형 번역이 목표라면 이 엔진이 핵심이다 (배치 STT 는 발화 끝까지 기다려야 함)

### TTS

**③ CosyVoice3 — 포트 8203**
- 2025-12-15 오픈소스. 모델 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
- `git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git`, **Python 3.10**
- `pip install -r requirements.txt` + `CosyVoice-ttsfrd` 리소스
- 양방향 스트리밍 TTS, first-token 지연 50% 개선, 3초 오디오로 zero-shot 음성 복제
- 서브모듈(`--recursive`) 누락이 흔한 실패 원인

**④ XTTS-v2 — 포트 8204**
- **원본 `coqui-ai/TTS` 대신 유지보수 포크를 쓸 것**: `pip install coqui-tts` ([idiap/coqui-ai-TTS](https://github.com/idiap/coqui-ai-TTS))
  import 경로는 동일 (`from TTS.api import TTS`), Python 3.10~3.14 + 최신 PyTorch 지원
- 17개 언어, 음성 복제
- 라이선스: 코드 MPL 2.0 / **가중치 CPML 비상업**. Coqui Inc. 가 2024-01 폐업해
  **상업 라이선스를 살 방법이 없다.** 상업화 시 교체 대상

**⑤ Qwen3-TTS 0.6B — 포트 8205**
- 2026-01-22 오픈소스. `pip install -U qwen-tts`
- 모델 2종:
  - `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (2.52GB) — 오디오 입력으로 빠른 음성 복제
  - `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` — 프리셋 음색 9종, 자연어 지시로 스타일 제어, 10개 언어
- 500만 시간 학습, 스트리밍 생성 지원

**⑥ Fish Speech — 포트 8206**
- [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech). 최신은 **S2** (2026-03)
- 라이선스: **FISH AUDIO RESEARCH LICENSE** — 연구·비상업 무료, 상업은 별도 유료 라이선스
  (XTTS-v2 와 달리 구매 경로는 존재한다)

---

## 3. GPU 환경에서 새로 고려할 것

**Docker GPU 접근 (WSL2)**
- Windows 호스트에 NVIDIA WSL2 드라이버, WSL 안에 NVIDIA Container Toolkit 필요
- compose 에 GPU 예약 추가:
  ```yaml
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  ```
- 베이스 이미지: `nvidia/cuda:12.x.x-cudnn-runtime-ubuntu22.04` (slim python 대신)

**VRAM 경합 — 이 구조에서 특히 중요**
- GPU 1장에 6개 엔진을 동시에 올리면 VRAM 이 터진다. CPU 서버에서보다 훨씬 심각하다
- 그래서 "필요한 것만 기동" 원칙이 여기서는 필수다
- `engine.env` 의 `PRELOAD=0` 을 적극 활용 (첫 요청까지 모델을 올리지 않음)
- 엔진별 VRAM 사용량을 측정해 `README` 표에 남길 것 (CPU 쪽에서 메모리·RTF 를 남긴 것처럼)

**아키텍처 차이**
- ysna-server 는 aarch64, WSL 은 x86_64. 이미지를 옮겨 쓸 수 없고 재빌드해야 한다
- x86_64 는 휠 가용성이 훨씬 좋아 ARM 에서 겪은 문제 대부분이 사라진다
  (참고: ARM 에서는 torch 기본 휠이 CUDA 빌드라 5GB 를 끌고 왔고, torchaudio 는 `+cpu`
  로컬버전 휠이 없어 `--no-deps` 로 넣어야 했다. x86_64 에서는 해당 없음)

---

## 4. 첫 단계 제안

1. `_common/`, `_template/`, `voicectl.sh` 를 WSL `~/deploy/voice` 로 가져온다
2. GPU·드라이버·Container Toolkit 확인 (`nvidia-smi`, `docker run --gpus all ... nvidia-smi`)
3. **whisper large-v3 부터** — 기존 `whisper/` 폴더를 거의 그대로 쓸 수 있어 구조 검증이 빠르다
4. 검증되면 CosyVoice3 → Qwen3-TTS → XTTS-v2 → Fish Speech 순으로 확장
5. Whisper streaming 은 WebSocket 규격 결정이 필요하므로 마지막에

각 엔진마다 컨테이너 이전에 `_common/mkvenv.sh <엔진>` 으로 호스트 venv 에서 먼저 검증하면
의존성 문제를 훨씬 빨리 잡는다 (ARM 쪽 작업에서 이 방식이 효과가 컸다).

---

## 5. CPU 서버(ysna-server) 현황 참고

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
