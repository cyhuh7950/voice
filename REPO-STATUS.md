## 2026-08-13 후속 — 병합 설계 완료, 커밋 전 (아래는 그 이후 기록)

§3-1(voiceapi.py/voicectl.sh 충돌)과 §3-2(.gitignore)는 **해결했다.** 아직 커밋은
하지 않았다 — 통합 방식(로컬 병합/PR/유지)을 사용자가 아직 정하지 않았다.

**출처가 확정됐다.** origin 의 4개 커밋은 이 세션이 만든 게 아니라 **ysna-server 에서
같은 사용자가 Opus 5 와 진행한 별도 세션**의 결과물이다 (작성자 `cyhuh7950
<echo_ai_dev@echoit.co.kr>`, `Co-Authored-By: Claude Opus 5`, translate 앱 연동
맥락). ysna-server 의 로컬 저장소를 읽기전용으로 확인한 결과 `HEAD == origin/main`
(56a5f8f) 이고 무관한 변경만 남아 있어, 기준이 더 움직일 위험 없이 안전하게
진행할 수 있었다.

**한 이유는 이렇다: `voiceapi.py` 는 애초에 병합할 필요가 없었다.** origin 이 이미
`create_app(routes=...)` 확장점을 만들어뒀고("한 엔진에만 필요한 라우트는 그 엔진이
직접 붙인다"), 이건 정확히 WebSocket 스트리밍 같은 단일 엔진 전용 기능을 위한
자리였다. 그래서:

- `_common/voiceapi.py` — **origin 것을 그대로 채택**(바이트 단위로 동일). 손대지 않았다
- `voicectl.sh` — origin 베이스 + WSL 전용 2줄(`docker compose` → `docker-compose`,
  이 서버에 v2 플러그인이 없어서)만 재적용
- `whisper-streaming/server.py` — `stream_factory=` 라는, origin 에 없는 파라미터를
  쓰던 옛 설계를 버리고 `routes=` 콜백으로 재작성. `StreamSession`/LocalAgreement-2
  로직은 전부 이 파일 안에 그대로 남았다(공유 파일을 안 건드리니 리스크가 크게 줄었다)

**추가로 발견해 고친 것 — `whisper/` 이름 충돌 (진짜 위험했던 부분):**
origin 의 `whisper/` = ysna 의 CPU 엔진(base, 8101). 로컬의 `whisper/` = 이 세션이
만든 ROCm GPU 엔진(large-v3, 8103, transformers 백엔드). **같은 폴더 이름으로
완전히 다른 두 엔진**을 가리키고 있었다 — 이 상태로 병합했으면 어느 쪽을 택해도
반대쪽 서버가 깨졌을 것이다. 해결: ROCm 엔진을 원래 그 용도로 있던 `whisper-gpu/`
폴더로 옮기고(죽은 NVIDIA 시절 내용을 대체, 모델 캐시 2.9GB 도 함께 이동해
재다운로드 없앰), `whisper/`는 origin 의 CPU 버전으로 복원했다. HANDOFF-GPU.md·
IMPROVEMENT.md 의 참조도 `whisper-gpu`로 맞춰 고쳤다.

**melotts/moonshine/supertonic 삭제도 철회했다.** 로컬에서(이 세션 이전에) 삭제돼
있었는데, origin 은 같은 시점에 그 파일들을 수정했다(TOTAL_STEPS 튜닝 등). 이 저장소는
ysna 와 히스토리를 공유하므로, 삭제를 커밋해 push 하면 ysna 가 다음 `git pull` 때
지금 운영 중인 파일을 잃는다. origin 버전으로 복원했다.

**README.md**: origin 버전(화자 엔진 포함, secrets.env 인증 설명)을 베이스로 하고,
로컬에만 있던 GPU 서버 섹션은 이제 다 틀린 정보라(RTX 5070/CUDA — GPU 가 AMD 로
교체됐다) 통째로 지우고 `HANDOFF-GPU.md` 를 가리키는 두 줄로 대체했다.
docker-compose 플러그인 부재 노트만 정확한 자리에 옮겨 살렸다.

**빌드 검증 (전부가 아니라 필요한 것만 개별로 — 원래 운용 방식 그대로):**

| 확인 대상 | 방법 | 결과 |
|---|---|---|
| whisper-gpu(8103) | 재빌드 후 기동, 배치 STT 왕복 | 원문과 완전 일치 (회귀 없음) |
| whisper-streaming(8104) | 재빌드 후 기동, 실제 WebSocket 클라이언트로 PCM 스트리밍 | LocalAgreement-2 partial→final 확정 정상 동작, 최종 텍스트 원문과 완전 일치 |

나머지 6개 GPU 엔진(cosyvoice·xtts·qwen3-tts·fish-speech)과 CPU 2개(zipformer·
moss-tts-nano)는 **재빌드하지 않았다** — voiceapi.py 가 origin 것과 완전히 동일하므로
동작이 바뀔 이유가 없고, "필요한 것만 개별로" 원칙상 지금 당장 검증할 필요가 없다.
다음에 그 엔진을 실제로 쓸 때(재기동 시 어차피 이미지가 새로 빌드된다) 확인하면 된다.
CPU 2개는 애초에 재빌드하지 않았고 계속 떠 있었다(영향 없음, voiceapi.py 를 그대로 쓰지만
쓰는 함수가 동일해 리스크 없음 — 다만 아직 실제로 새 이미지로 재기동해보진 않았다).

**`git add` 시뮬레이션(`-n`, 실행 안 함)으로 찾은 추가 위험:**

`cosyvoice/repo`, `fish-speech/repo`, `moss-tts-nano/repo` 세 곳이 **embedded git
repository(gitlink)로 잡힌다.** 이대로 커밋하면 이 폴더들은 실제 파일 없이 커밋 SHA
참조만 남고, 다른 곳에서 `git clone` 하면 **빈 폴더로 받아진다** — Docker 빌드가
전부 깨진다. (비밀값/대용량 파일은 새로 추가된 `.gitignore` 가 82개 항목을 정상적으로
걸러내는 것을 확인했다 — 이 문제는 아니다.)

두 가지 선택지가 있고 트레이드오프가 있어 결정하지 않았다:

- **A. 정식 git submodule 로 전환** — IMPROVEMENT.md 가 원래 "업스트림 저장소를
  git 이력째 들고 있는다"고 적어둔 것과 맞다. 다만 clone 할 때 `--recurse-submodules`
  를 잊으면 조용히 빈 폴더가 되고, 벤더링한 소스를 직접 고치는 이번 프로젝트의
  핵심 워크플로(IMPROVEMENT.md ⑤ "벤더링된 저장소 직접 수정")와는 안 맞는다 — 로컬
  수정이 서브모듈 안쪽의 "dirty" 상태가 되어 이 저장소의 커밋만으로는 안 남는다
- **B. 중첩 `.git` 을 지우고 평범한 파일로 추적** — 업스트림 커밋 이력은 잃지만,
  벤더링한 소스를 고치는 작업이 이 저장소의 평범한 커밋이 된다 (서브모듈 관리 불필요).
  IMPROVEMENT.md 의 실제 강조점(직접 수정)과는 이쪽이 더 맞는다

**2026-08-25 — 위 두 결정이 났다. 처리 완료:**

1. **API_KEY → `secrets.env` 로 통일 (결정: 분리).** 8개 GPU 엔진
   (whisper-gpu/whisper-streaming/cosyvoice/xtts/qwen3-tts/fish-speech/zipformer/
   moss-tts-nano) 전부에서 `engine.env` 의 `API_KEY=` 줄을 지우고,
   `secrets.env.example` 을 origin/ysna 와 동일한 문구로 추가했으며,
   `docker-compose.yml` 의 `env_file` 에 `secrets.env`(`required: false`) 를
   덧붙였다. `_template/engine.env` 도 같은 방식으로 고쳤다 — origin 자체가
   `_template` 은 아직 이 패턴으로 안 옮겨놓은 상태였다.
   **실제 키 값은 만들지 않았다** — 지금처럼 비워두고, 채우고 싶으면
   `cp secrets.env.example secrets.env && chmod 600 secrets.env` 후 채우면 된다.

2. **vendored repo 세 개 → B안(중첩 `.git` 제거) 채택.** `cosyvoice/repo`,
   `fish-speech/repo`, `moss-tts-nano/repo` 안의 `.git` 을 지우고 평범한 파일로
   바꿨다. 원본 출처는 각 폴더의 `repo/VENDORED.md` 에 남겼다(URL·벤더링 시점
   커밋 해시·시각). `git add -A -n` 재확인 결과 "embedded git repository" 경고가
   사라졌다.

**남은 결정은 이제 하나뿐이다: 통합 방식** (로컬 병합 / PR / 지금 상태 유지).
그 외 내용상 반영해야 할 변경은 더 없다 — 커밋 직전 상태다.

**git 조작 기록:** `git checkout origin/main -- <path>` 로 melotts/moonshine/
supertonic/whisper/_template 일부를 가져왔고, `_common/voiceapi.py` 는 origin 내용을
그대로 덮어썼다. **커밋은 하지 않았다.** push 도 하지 않았다.

---

# 저장소 통합 대기 상태 (2026-08-13 기준, 최초 작성분 — 위 후속 기록 이후로는 일부 항목이 해결됨)

이 문서는 저장소 통합·정리 절차를 진행하다 **중단 조건에 걸려 대기 중**임을 기록한다.
다음 작업자(사람이든 세션이든)는 아래를 먼저 읽고, git 조작 전에 사용자 확인을 받을 것.

**현재 git 조작 없음 — `main`은 그대로다.** 이 문서 자체도 커밋되지 않은 신규 파일이다.

---

## 1. 왜 멈췄는가

통합 절차(완료 검증 → 통합 방식 결정 → 실행)를 시작했으나, **1단계 완료 검증에서
중단 조건 4가지가 동시에 걸려** 사용자에게 보고했고, 사용자는 "현재 상태 유지"를
선택했다(2026-08-13).

## 2. 저장소 정체성 (확인됨)

| 항목 | 값 |
|---|---|
| 경로 | `/home/daon/deploy/voice` |
| 현재 브랜치 | `main` (별도 작업 브랜치 없이 처음부터 여기서 직접 작업) |
| 작업공간 유형 | 일반 저장소 (worktree 아님) |
| 원격 | `origin` → `github-cyhuh7950:cyhuh7950/voice.git` |
| 공식 검증 명령 | **없음** — test/lint/build 설정 파일이 저장소에 없다.
검증은 각 엔진 컨테이너 기동 + HTTP 왕복 테스트로 대신했다 (아래 §5) |

## 3. 중단 원인 — 반드시 먼저 해결할 것

### 3-1. 로컬 `main`이 `origin/main`보다 4개 커밋 뒤처져 있고, 겹치는 파일이 있다

```
origin/main 에만 있는 커밋 (fetch 완료, 2026-08-13 확인):
  56a5f8f restart 가 새 이미지를 쓰도록 고친다
  6519533 사용자 노출 메시지를 영어로 전환
  de5aeaf speaker 엔진 — 화자 임베딩 (ONNX, 8301)
  ae1db08 supertonic: TOTAL_STEPS 8 → 6
```

merge-base 는 로컬 HEAD(`a3bff3f`)와 같다 — 즉 히스토리가 갈라진 게 아니라
**로컬이 단순히 4커밋 뒤처진 것**뿐이다. fast-forward pull 자체는 가능하다.

문제는 그 4개 커밋이 건드린 파일과 **로컬 미커밋 변경이 정면으로 겹친다는 것**:

| 파일 | origin 의 변경 | 로컬(WSL)의 미커밋 변경 |
|---|---|---|
| `_common/voiceapi.py` | 534줄 변경 | WebSocket 스트리밍 엔드포인트 추가 등 대폭 수정 |
| `voicectl.sh` | 194줄 변경 | 수정 |
| `melotts/`, `moonshine/`, `supertonic/` | server.py·engine.env 수정 | **폴더째 삭제(D)** |
| `whisper/` | server.py 수정 | ROCm 용으로 전면 재작성 |
| `speaker/` (신규) | 8301 포트로 추가 | 로컬에 없음 |

**`melotts`/`moonshine`/`supertonic`이 핵심이다.** 로컬은 이 서버를 GPU 전용으로
쓰기로 하고 이 세 엔진을 지웠는데, origin 은 같은 시점에 그 파일들을 고쳤다.
이건 자동 병합이 안전하지 않은 **삭제 vs 수정 충돌**이다.

→ 사용자 확인 필요: **이 세 엔진을 이 저장소(WSL 쪽 브랜치)에서 계속 삭제 상태로
유지할지, origin 의 최신 수정을 받아들인 뒹 다시 삭제할지, 아니면 애초에 삭제하면
안 됐던 것인지.**

**출처 확인됨 (2026-08-13):** 이 4개 커밋은 이 세션(WSL, Sonnet 5)이 만든 게 아니다.
전부 작성자 `cyhuh7950 <echo_ai_dev@echoit.co.kr>`, `Co-Authored-By: Claude Opus 5`.
**ysna-server 에서 진행된 별도 세션의 결과물**이다 — 커밋 내용 자체가 그 증거다
(`de5aeaf`: 번역 앱 연동용 speaker 엔진, `ae1db08`: ARM 4코어 실측 튜닝,
`56a5f8f`: "translate 저장소의 translatectl.sh 도 같은 문제였다" — translate 라는
별개 저장소까지 언급). ysna-server 는 자기 자신의 `~/deploy/voice` 클론을 갖고
있고 그 쪽에서 이 origin 에 먼저 push 한 것으로 보인다. 즉 이건 실수나 사고가
아니라 **두 서버에서 같은 저장소를 각자 세션으로 동시에 발전시켜 온 결과**다.

### 3-2. `.gitignore`가 로컬에서 삭제된 채 커밋되지 않았다

```
D .gitignore
```

이 파일이 `secrets.env`, `*/models/`, `*/.venv/`, `__pycache__/`를 제외 대상으로
지정하고 있었다. **삭제 경위가 대화 이력에 없어 원인 불명 — 실수인지 의도인지
사용자에게 확인해야 한다.**

지금 상태로 `git add -A` 를 하면 안 되는 이유:
- `.venv`/`venv` 수 GB (cosyvoice, whisper-gpu, edge-tts)
- 모델 캐시 40GB+ (cosyvoice 18GB, fish-speech 11GB, qwen3-tts 4.7GB, whisper-gpu 3.5GB, whisper 2.9GB 등)
- `__pycache__`
- **중첩 `.git` 3개**: `cosyvoice/repo/.git`, `fish-speech/repo/.git`, `moss-tts-nano/repo/.git`
  (업스트림을 벤더링하며 clone 한 것들. `.gitignore` 없이 `git add` 하면 gitlink 로
  잘못 스테이징되거나 예기치 않게 처리될 수 있다)

`secrets.env` 류의 `API_KEY` 실값은 확인 결과 **전부 공란**이라 즉시 유출 위험은
없다. 하지만 값을 채운 뒤 이 상태로 커밋하면 위험해진다.

### 3-3. "승인된 작업 범위"를 판단할 근거 문서가 저장소 안에 없다

이번 세션에서 무엇을 했는지는 대화 이력에 있지만, 저장소 자체에는 이슈/계획
문서가 없다. 47개 변경/미추적 항목이 전부 범위 안이라는 판단은 **대화 맥락에
근거한 추정이지 문서로 확정된 사실이 아니다.**

### 3-4. 미검증 엔진 2종

| 엔진 | 포트 | 상태 |
|---|---|---|
| whisper-streaming | 8104 | 코드·이미지(16.8GB) 완성, **WebSocket 동작 검증 미완료** |
| fish-speech | 8206 | `BytesIO` 버그 수정 완료, **재빌드·합성 검증 미완료** (디스크 여유 대기 중) |

## 4. 지금 git 상태 스냅샷 (참고용, 시간이 지나면 stale 해진다)

```
HEAD: a3bff3f (main)
origin/main: 56a5f8f (로컬보다 4커밋 앞)

수정(M): .dockerignore, HANDOFF-GPU.md, README.md, _common/voiceapi.py,
         _template/Dockerfile, _template/docker-compose.yml, voicectl.sh,
         whisper/{Dockerfile,docker-compose.yml,engine.env,requirements.txt,server.py}

삭제(D): .gitignore, melotts/*, moonshine/*, supertonic/*, whisper/secrets.env.example

미추적(??): IMPROVEMENT.md, REPO-STATUS.md(이 파일), _base/, _common/__pycache__/,
            cosyvoice/, edge-tts/, fish-speech/, moss-tts-nano/, qwen3-tts/,
            whisper-gpu/, whisper-streaming/, whisper/__pycache__/, whisper/models/,
            work_order_wsl_rocm.md, xtts/, zipformer/
```

## 5. 이번 세션에서 실제로 검증한 것 (공식 테스트가 없어 이것이 유일한 근거)

| # | 엔진 | 포트 | 장치 | RTF | 검증 |
|---|---|---|---|---|---|
| ① | whisper large-v3 | 8103 | GPU | 0.338 | 한국어 인식 정확 일치 |
| ② | whisper-streaming | 8104 | GPU | - | **미검증** |
| ③ | cosyvoice3 | 8203 | GPU | 1.700 | 합성→whisper 왕복 통과 |
| ④ | xtts-v2 | 8204 | GPU | 3.362 | 합성→whisper 왕복 통과 (참조 음성 6초 미만 시 붕괴 주의) |
| ⑤ | qwen3-tts | 8205 | GPU | 6.500 | 합성→whisper 왕복 통과 |
| ⑥ | fish-speech | 8206 | GPU | - | **미검증** (KV 캐시 4096 상한 적용, BytesIO 패치 완료) |
| ⑦ | zipformer | 8105 | CPU | 0.010(WSL)/0.05~0.14(ysna) | 인식 되나 정확도 낮음. 띄어쓰기는 수정 완료 |
| ⑧ | moss-tts-nano | 8207 | CPU | 0.541(WSL)/1.1(ysna) | 참조 음성 복제로 왕복 완전 일치 |

ysna-server(별도 저장소, 이 저장소와 무관)에도 zipformer·moss-tts-nano 2종을
추가 설치·검증 완료했다 (WSL 저장소와는 별개 작업, 이 문서 범위 밖).

## 6. 다음 작업자가 할 일 — 순서대로

1. **사용자에게 §3-1, §3-2 를 확인받는다.** 특히 `.gitignore` 삭제 경위와
   melotts/moonshine/supertonic 처리 방향.
2. 확인되면:
   - `.gitignore` 복원 또는 재작성
   - `git fetch && git merge origin/main` (또는 rebase — 사용자 지시에 따름)
   - `_common/voiceapi.py`, `voicectl.sh` 충돌을 손으로 해소 (양쪽 변경 모두 보존)
   - melotts/moonshine/supertonic 충돌을 사용자 결정대로 처리
3. 병합 후 **8개 GPU 엔진 전부 재기동해 회귀 확인** (voiceapi.py 가 공유 계약이라
   병합 실수 하나가 8개 전부를 깰 수 있다)
4. ②⑥ 미검증 엔진 마저 검증 (GPU 단독 기동 필요 — §"운용 방침" 참고, HANDOFF-GPU.md)
5. 그제서야 커밋 → 통합 방식(로컬 병합/PR/유지) 재논의

## 7. 절대 하지 말 것

- `.gitignore` 없는 상태에서 `git add -A`
- `origin/main` 을 무시하고 강제 push (다른 세션/작업자의 결과물을 파괴함)
- `melotts`/`moonshine`/`supertonic`/`.gitignore` 를 임의로 복원하거나 확정 짓기
  (사용자 확인 전)
