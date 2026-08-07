# voice — STT / TTS 엔진 컨테이너

STT·TTS 오픈소스 엔진을 **각각 독립된 컨테이너**로 띄우고, 외부에서 **HTTP API**로 호출하는 구성이다.
엔진끼리 의존성이 없어 필요한 것만 골라 켜고, 새 엔진은 폴더 하나 추가로 붙는다.

```
voice/
├── voicectl.sh          제어 스크립트 (폴더를 자동 탐색한다)
├── _common/
│   ├── voiceapi.py      모든 엔진이 공유하는 HTTP 레이어
│   └── mkvenv.sh        폴더별 가상환경 생성
├── _template/           새 엔진 추가용 템플릿
│
├── whisper/             STT  8101   ┐
├── moonshine/           STT  8102   │ 각 폴더가 완결된 서비스 단위:
├── supertonic/          TTS  8201   │ engine.env, requirements.txt,
└── melotts/             TTS  8202   ┘ server.py, Dockerfile, compose, .venv, models
```

포트 규칙: **STT는 81xx, TTS는 82xx**.

---

## 엔진 현황 (이 서버 실측)

ARM64(Neoverse-N1) 4코어 · GPU 없음 · CPU 추론 기준.
RTF(Real-Time Factor)는 `처리시간 ÷ 오디오길이`로, **1보다 작으면 실시간보다 빠르다**.

| 엔진 | 포트 | 종류 | 모델 | 메모리 | RTF | 언어 |
|---|---|---|---|---|---|---|
| whisper | 8101 | STT | base / int8 | 323 MB | **1.12** | 99개 자동 감지 |
| moonshine | 8102 | STT | tiny-ko | 216 MB | **0.05 ~ 0.18** | 8개 (컨테이너당 1개) |
| supertonic | 8201 | TTS | supertonic-3 | 487 MB | **1.93** | 31개, 보이스 10종 |
| melotts | 8202 | TTS | MeloTTS-KR | 약 1.5 GB | **약 6** | 6개 (컨테이너당 1개) |

### 엔진 선택 가이드

- **음성 대화형 번역이 목적이면 `whisper` + `supertonic` 조합**을 권한다.
  한국어 인식이 정확하고(whisper), 합성이 다국어 한 모델로 되며 충분히 빠르다(supertonic).
- `moonshine`은 **영어 실시간 인식**에 강하다. RTF 0.05는 압도적으로 빠르다.
  다만 한국어는 tiny 모델뿐이고 **문장 끝 단어를 자주 놓친다** (아래 "알려진 제약" 참고).
- `melotts`는 이 CPU에서 RTF 약 6이다. 3초 문장에 20초가 걸려 **실시간 대화용으로는 부적합**하다.
  품질 비교나 배치 합성용으로 두는 편이 맞다.

---

## 사용법

엔진 목록과 상태:

```bash
./voicectl.sh list
```

필요한 것만 띄운다 (이미지가 없으면 자동으로 빌드한다):

```bash
./voicectl.sh start whisper supertonic
```

`start`에 인자를 주지 않으면 아무것도 뜨지 않는다. **전체를 한꺼번에 올리는 명령은 없다** —
모든 엔진을 메모리에 상주시키지 않는 것이 이 구성의 기본 방침이다.

```bash
./voicectl.sh status              # 상태 / 헬스 / 메모리 / 응답
./voicectl.sh stop melotts        # 지정한 것만 정지
./voicectl.sh logs whisper -f     # 로그 추적
./voicectl.sh test supertonic "안녕하세요"          # TTS 합성 왕복 테스트
./voicectl.sh test whisper /tmp/voicectl-supertonic.wav   # STT 인식 왕복 테스트
```

---

## API

4개 엔진 모두 같은 규격을 쓴다. **OpenAI Audio API와 호환**되므로 기존 클라이언트를 그대로 붙일 수 있다.
브라우저에서 온 webm/opus, mp3, m4a, flac 등은 서버가 ffmpeg으로 알아서 변환한다.

각 엔진의 `GET /docs`에 대화형 OpenAPI 문서가 있다.

### 공통

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | 상태·모델 로딩 여부. 인증 불필요. 로딩 중이면 503 |
| GET | `/info` | 엔진 정보, 지원 언어, 엔드포인트 목록 |
| GET | `/docs` | OpenAPI 문서 |

### STT — whisper(8101), moonshine(8102)

```bash
# 기본
curl -F "file=@audio.wav" http://localhost:8101/v1/audio/transcriptions

# 언어 지정 + 구간 정보까지
curl -F "file=@audio.wav" -F language=ko -F response_format=verbose_json \
     http://localhost:8101/v1/audio/transcriptions

# 텍스트만 받기
curl -F "file=@audio.wav" -F response_format=text \
     http://localhost:8101/v1/audio/transcriptions
```

```jsonc
{
  "text": "안녕하세요 음성 번역 테스트입니다",
  "language": "ko",
  "duration": 3.344,          // 오디오 길이(초)
  "engine": "whisper",
  "processing_s": 3.649       // 처리 시간(초)
}
```

폼 필드: `file`(필수), `language`, `prompt`, `temperature`, `response_format`(`json`|`text`|`verbose_json`)

`POST /v1/audio/translations`는 **음성을 영어 텍스트로** 바로 번역한다 (whisper만 지원).
다만 base 모델의 번역 품질은 낮다. 번역은 LLM에 맡기고 STT는 원문 인식만 시키는 쪽이 낫다.

### TTS — supertonic(8201), melotts(8202)

```bash
# 보이스 목록
curl http://localhost:8201/v1/voices

# 합성
curl -H 'Content-Type: application/json' \
     -d '{"input":"오라클 서버에서 음성 합성이 동작합니다.","voice":"F1","language":"ko"}' \
     -o out.wav http://localhost:8201/v1/audio/speech

# mp3 로 받기
curl -H 'Content-Type: application/json' \
     -d '{"input":"Hello there.","voice":"M1","language":"en","response_format":"mp3"}' \
     -o out.mp3 http://localhost:8201/v1/audio/speech
```

JSON 필드: `input`(필수), `voice`, `language`, `speed`(0.25~4.0),
`response_format`(`wav`|`mp3`|`opus`|`flac`|`pcm`)

응답은 오디오 바이트이고, 참고 정보는 헤더로 온다:
`X-Engine`, `X-Sample-Rate`, `X-Audio-Duration`, `X-Processing-Seconds`

### 파이썬에서 (OpenAI SDK 그대로)

```python
from openai import OpenAI

stt = OpenAI(base_url="http://<서버>:8101/v1", api_key="unused")
with open("audio.wav", "rb") as f:
    text = stt.audio.transcriptions.create(model="whisper", file=f).text

tts = OpenAI(base_url="http://<서버>:8201/v1", api_key="unused")
audio = tts.audio.speech.create(model="supertonic", voice="F1", input=text)
audio.write_to_file("out.wav")
```

`API_KEY`를 설정했다면 `api_key`에 그 값을 넣는다.

---

## 외부 노출 — nginx-proxy-manager 경유

이 서버들의 공통 규약대로, 모든 엔진 컨테이너는 **`proxy-network`** 에 붙는다.
nginx-proxy-manager(NPM)와 같은 네트워크이므로 NPM이 **컨테이너 이름으로 바로 forward** 한다.

```
인터넷 ──443──▶ nginx-proxy-manager ──▶ voice-whisper:8101   (proxy-network 내부)
                  (TLS 처리)             voice-moonshine:8102
                                         voice-supertonic:8201
                                         voice-melotts:8202
```

엔진별로 VCN 보안 목록에 포트를 따로 열 필요가 없다. **443은 이미 열려 있고 NPM이 쓰고 있다.**

### 공개 엔드포인트 (등록 완료)

| 도메인 | Forward | 엔진 |
|---|---|---|
| `https://stt.whisper.sinsan.kr` | `voice-whisper:8101` | whisper (STT) |
| `https://stt.moonshine.sinsan.kr` | `voice-moonshine:8102` | moonshine (STT) |
| `https://tts.supertonic.sinsan.kr` | `voice-supertonic:8201` | supertonic (TTS) |
| `https://tts.melotts.sinsan.kr` | `voice-melotts:8202` | melotts (TTS) |

NPM Proxy Host 설정: Scheme `http`, Forward Hostname은 **컨테이너 이름**,
Forward Port는 컨테이너 내부 포트, SSL은 Let's Encrypt + Force SSL. Websockets는 불필요.

Forward Port를 틀리면 502가 난다. 도달 확인은 NPM 컨테이너 안에서 직접 하는 게 가장 빠르다:

```bash
docker exec nginx-proxy-manager curl -s http://voice-whisper:8101/health
```

외부에서 확인:

```bash
curl https://stt.whisper.sinsan.kr/health          # 인증 불필요
curl -H "Authorization: Bearer <키>" -F "file=@audio.wav" \
     https://stt.whisper.sinsan.kr/v1/audio/transcriptions
```

### 인증 (적용됨)

네트워크 경로와 인증은 별개다. NPM에 등록하면 그 도메인은 인터넷에 열리므로 앱 레벨 인증이 필요하다.
**키는 `<엔진>/secrets.env` 에 있다.** 엔진마다 다른 키를 쓰므로 개별 회수·교체가 가능하다.
이 파일은 버전관리에서 제외되므로, 저장소를 clone 한 서버에서는 직접 만들어야 한다.

```bash
grep '^API_KEY=' whisper/secrets.env             # 키 확인 (파일 권한 600)

# clone 직후 — 엔진마다 한 번씩
cp whisper/secrets.env.example whisper/secrets.env
chmod 600 whisper/secrets.env
# 그리고 값을 채운다. 새로 만들려면:
openssl rand -hex 24
```

`secrets.env` 가 없어도 컨테이너는 뜬다 (`required: false`). 다만 그 경우 인증이 꺼진 채로
동작하므로, 외부에 노출한다면 반드시 채울 것.

```bash
curl -H "Authorization: Bearer <키>" ...
# 또는 -H "X-API-Key: <키>"
```

`/health`만 인증 없이 열려 있다 (로드밸런서·모니터링용). 그 외 엔드포인트는 키 없이 401이다.
`voicectl.sh test`는 `engine.env`에서 키를 읽어 자동으로 붙인다.

키를 바꾸려면 `secrets.env`를 고치고 해당 엔진만 재시작하면 된다. 인증을 끄려면 `API_KEY=`로 비운다.
NPM의 Access List(basic auth / IP 허용목록)를 추가로 걸어도 된다.

### 호스트 포트 바인딩 (적용됨)

`BIND_HOST=127.0.0.1`이라 호스트 포트는 **루프백에만** 열려 있다.

```
127.0.0.1:8101  127.0.0.1:8102  127.0.0.1:8201  (:8202)
```

서버 내부에서 `curl localhost:8101`은 되고, 외부에서 포트를 직접 때리는 경로는 없다.
NPM은 `proxy-network`로 컨테이너에 직접 붙으므로 이 설정과 무관하게 동작한다 (검증됨).

호스트 방화벽(iptables INPUT)은 정책 ACCEPT로 차단 룰이 없다. 즉 이전에 포트가 외부에 닿지 않은 건
Oracle Cloud VCN 보안 목록 덕분이었고, 지금은 그 계층에 의존하지 않는다.
포트를 직접 노출해야 할 일이 생기면 `BIND_HOST=0.0.0.0`으로 바꾸고 VCN도 함께 열어야 한다.

---

## 새 엔진 추가

```bash
./voicectl.sh new piper tts 8203
```

`_template`을 복사해 이름·종류·포트를 치환한 폴더가 만들어진다. 그다음 3가지만 채우면 된다.

1. **`requirements.txt`** — 파이썬 의존성.
   이 서버는 aarch64라서 **휠이 있는지 먼저 확인**할 것. 소스 빌드로 넘어가면 매우 오래 걸리거나 실패한다.
   ```bash
   curl -s https://pypi.org/pypi/<패키지>/json \
     | python3 -c "import sys,json; d=json.load(sys.stdin); v=d['info']['version']; print([f['filename'] for f in d['releases'][v]])"
   ```

2. **`server.py`** — `load()`와, STT면 `transcribe()` / TTS면 `synthesize()`.
   HTTP 규격·오디오 변환·인증·health는 `_common/voiceapi.py`가 처리하므로 손대지 않는다.
   ```python
   # STT: 16kHz mono float32 numpy → dict
   def transcribe(audio, *, language, prompt, temperature, task) -> dict:
       return {"text": "...", "language": "ko", "segments": []}

   # TTS: 텍스트 → (mono float32 numpy, sample_rate)
   def synthesize(text, *, voice, language, speed) -> tuple[np.ndarray, int]:
       return wav, 24000
   ```

3. **`engine.env`** — 엔진별 설정. `voicectl.sh`가 이 파일로 서비스를 인식한다.

컨테이너로 넘어가기 전에 호스트에서 먼저 검증하는 쪽이 빠르다:

```bash
./_common/mkvenv.sh piper        # piper/.venv 생성 + 설치
piper/.venv/bin/python piper/server.py
```

`server.py`는 컨테이너(`/app/voiceapi.py`)와 호스트(`../_common/voiceapi.py`) 양쪽에서 그대로 돈다.

---

## 폴더별 가상환경

의존성 충돌을 막기 위해 격리를 두 겹으로 둔다.

- **호스트**: `<엔진>/.venv` — `_common/mkvenv.sh <엔진>`으로 생성. 컨테이너 없이 바로 돌려보거나 디버깅할 때.
- **이미지**: `/opt/venv` — Dockerfile이 같은 `requirements.txt`로 동일하게 만든다.

`ffmpeg`은 오디오 변환에 반드시 필요하다. 이미지에는 들어 있고, 호스트 venv로 직접 돌릴 때는 호스트에 있어야 한다.

모델 캐시는 `<엔진>/models/`에 남아 컨테이너를 다시 만들어도 재다운로드하지 않는다.
현재 캐시 크기: whisper 142MB, moonshine 360MB, supertonic 386MB, melotts 1.2GB.

---

## 알려진 제약

**moonshine 한국어가 문장 끝을 놓친다.** 실측:

| 원문 | moonshine(tiny-ko) | whisper(base) |
|---|---|---|
| 오라클 서버에서 음성 합성이 동작합니다. | 오라클 서버에서 음성 합성이 | 오라클 서버에서 음성 합성이 동작합니다. |
| 안녕하세요, 음성 번역 테스트입니다. | 안녕하세요. 음성 번역 | 안녕하세요 음성 번역 테스트입니다 |

배치 / 스트리밍 / `FORCE_UPDATE` 세 경로 모두 같았고 무음 패딩도 효과가 없었다.
같은 파일을 whisper가 완전히 인식하므로 오디오 문제가 아니라 **tiny-ko 모델의 한계**다.
한국어는 tiny 모델만 공개돼 있어 우회할 방법이 없다. 영어(tiny/base)는 같은 문장을 완전히 인식했다.
→ **한국어 인식은 whisper(8101)를 쓸 것.**

**melotts가 느리다.** ARM CPU에서 RTF 약 6. 실시간 대화에는 supertonic을 쓸 것.

**컨테이너 하나 = 언어 하나** (moonshine, melotts). 모델이 언어별로 분리돼 있어서다.
다국어가 필요하면 `engine.env`의 언어를 바꿔 재시작하거나, 언어별로 폴더를 복제해 포트를 달리 준다.

**moonshine 캐시 경로는 절대경로여야 한다.** `MOONSHINE_VOICE_CACHE`에 상대경로를 주면
모델을 받아놓고도 못 찾는다. 컨테이너는 `/models`라 해당 없고, 호스트 venv로 돌릴 때만 주의.

**라이선스.** 상업적으로 쓸 계획이라면 확인이 필요하다.

| 엔진 | 코드 | 모델 |
|---|---|---|
| whisper | MIT (faster-whisper) | MIT (OpenAI Whisper) |
| moonshine | MIT | **Moonshine Community License — 비상업용.** 기동 시 로그에도 경고가 찍힌다 |
| supertonic | 업스트림 저장소 확인 필요 | 업스트림 저장소 확인 필요 |
| melotts | MIT | MIT |

---

## 참고

- 첫 기동은 모델 다운로드로 수 분 걸린다. `/health`가 `"ready": true`가 될 때까지 503을 돌려준다.
- `PRELOAD=0`으로 두면 첫 요청까지 모델을 올리지 않아 메모리를 아낀다 (첫 응답은 느려진다).
- 모델을 크게 바꾸려면 `engine.env`의 `WHISPER_MODEL`(`tiny`~`large-v3`)이나 `COMPUTE_TYPE`을 조정한다.
- 여러 엔진을 동시에 돌리면 4코어를 나눠 쓰게 된다. `CPU_THREADS`, `INTRA_OP_THREADS`, `OMP_NUM_THREADS`로 제한할 수 있다.
