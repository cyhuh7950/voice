"""
whisper-streaming (stt) 서버 — Whisper large-v3 스트리밍, GPU(AMD ROCm/HIP).

배치 STT(8103)는 발화가 끝나야 결과가 나온다. 대화형 용도에는 그 지연이 치명적이라
말하는 도중에 중간 결과를 계속 내보내는 엔진이 따로 필요하다.

**백엔드가 기존 구현체와 다르다.** WhisperLive / whisper_streaming 은 둘 다
faster-whisper(CTranslate2)를 전제하는데, CTranslate2 는 ROCm 백엔드가 없어 AMD GPU 에서
GPU 가속을 못 쓴다. 그래서 배치 엔진(whisper 8103)에서 검증된 transformers 백엔드를
그대로 재사용하고, 확정 정책만 여기서 구현한다.

확정 정책은 whisper_streaming 의 **LocalAgreement-2** 를 따른다.
  연속한 두 번의 추론이 접두사(prefix)로 일치하는 부분까지만 "확정(final)"으로 내보내고,
  나머지는 "잠정(partial)"으로 둔다. 다음 오디오가 붙으면 뒤쪽은 얼마든지 바뀔 수 있기 때문이다.
이렇게 하면 확정된 텍스트는 뒤집히지 않는다 (사용자 화면에서 글자가 되돌아가지 않는다).

HTTP/WS 규격은 _common/voiceapi.py 가 처리한다. 여기서는 모델 로딩과 추론만 구현한다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

# ROCm(WSL) 필수 환경변수. 이미지 ENV 에도 있지만 호스트 venv 로 직접 돌릴 때를 위해 여기서도 세운다.
os.environ.setdefault("HSA_ENABLE_DXG_DETECTION", "1")
os.environ.setdefault("ROCPROFILER_REGISTER_ENABLED", "0")

_here = Path(__file__).resolve().parent
sys.path[:0] = [str(_here), str(_here.parent / "_common")]

import numpy as np  # noqa: E402
from fastapi import WebSocket, WebSocketDisconnect  # noqa: E402

from voiceapi import STT_SAMPLE_RATE, EngineSpec, create_app, run  # noqa: E402

log = logging.getLogger("whisper-streaming")

MODEL = os.getenv("MODEL_NAME", "openai/whisper-large-v3")
MODEL_CACHE = os.getenv("MODEL_CACHE", "/models")
DEVICE = os.getenv("DEVICE", "cuda")          # ROCm 에서도 torch API 는 "cuda" 다 (HIP 매핑)
DTYPE = os.getenv("COMPUTE_TYPE", "float16")
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "1"))  # 스트리밍은 지연이 우선이라 기본 greedy
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "") or None

# 이만큼 새 오디오가 쌓일 때마다 한 번 추론한다. 짧을수록 반응이 빠르고 GPU 부담이 커진다.
CHUNK_SECONDS = float(os.getenv("CHUNK_SECONDS", "1.0"))
# 확정된 뒤 버퍼에서 잘라낼 기준. whisper 는 30초 창을 쓰므로 그 안쪽으로 유지해야 한다.
MAX_BUFFER_SECONDS = float(os.getenv("MAX_BUFFER_SECONDS", "25.0"))

_model = None
_processor = None
_torch = None

_LANG_TOKEN = re.compile(r"<\|([a-z]{2,3})\|>")


def load() -> None:
    global _model, _processor, _torch
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    _torch = torch
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[DTYPE]

    _processor = AutoProcessor.from_pretrained(MODEL, cache_dir=MODEL_CACHE)
    _model = WhisperForConditionalGeneration.from_pretrained(
        MODEL,
        cache_dir=MODEL_CACHE,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",   # flash_attention_2 는 gfx1101 휠이 없다
    ).to(DEVICE)
    _model.eval()


def _detect_language(features) -> str | None:
    ids = _model.detect_language(features)
    token = _processor.tokenizer.convert_ids_to_tokens(int(ids.flatten()[0]))
    m = _LANG_TOKEN.fullmatch(token)
    return m.group(1) if m else None


def _decode(audio: np.ndarray, language: str | None) -> tuple[str, str | None]:
    """버퍼 전체를 한 번 디코딩한다. (텍스트, 감지언어)"""
    torch = _torch
    dtype = next(_model.parameters()).dtype

    # 스트리밍 버퍼는 항상 30초 미만으로 유지하므로 short-form 경로만 쓴다.
    inputs = _processor(audio, sampling_rate=STT_SAMPLE_RATE, return_tensors="pt")
    features = inputs.input_features.to(DEVICE, dtype=dtype)

    lang = language or _detect_language(features)
    gen_kwargs: dict = {"task": "transcribe", "num_beams": BEAM_SIZE}
    if lang:
        gen_kwargs["language"] = lang

    with torch.inference_mode():
        out = _model.generate(features, **gen_kwargs)
    text = _processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    return text, lang


def _common_prefix(a: str, b: str) -> str:
    """두 문자열의 공통 접두사. 단어 중간에서 자르지 않도록 마지막 공백까지만 인정한다."""
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    if n == len(a) == len(b):
        return a
    cut = a.rfind(" ", 0, n)
    return a[:cut] if cut > 0 else ""


class _Session:
    """LocalAgreement-2 기반 스트리밍 세션. voiceapi.StreamSession 규격을 따른다."""

    def __init__(self, language: str | None = None):
        self.language = language or DEFAULT_LANGUAGE
        self._buf = np.zeros(0, dtype=np.float32)   # 아직 확정되지 않은 구간의 오디오
        self._pending = 0                            # 마지막 추론 이후 쌓인 샘플 수
        self._prev = ""                              # 직전 추론 결과 (LocalAgreement 비교용)
        self._confirmed = ""                         # 이번 버퍼에서 확정된 부분
        self._offset = 0.0                           # 버퍼 시작 시각(초). 잘라낸 만큼 누적된다

    # -- voiceapi.StreamSession -------------------------------------------

    def feed(self, audio: np.ndarray) -> list[dict]:
        self._buf = np.concatenate([self._buf, audio])
        self._pending += audio.size
        if self._pending < CHUNK_SECONDS * STT_SAMPLE_RATE:
            return []
        self._pending = 0
        return self._step(final=False)

    def finish(self) -> list[dict]:
        if self._buf.size == 0:
            return []
        return self._step(final=True)

    # -- 내부 --------------------------------------------------------------

    def _step(self, *, final: bool) -> list[dict]:
        text, lang = _decode(self._buf, self.language)
        if lang and not self.language:
            self.language = lang

        events: list[dict] = []

        if final:
            # 마지막이므로 남은 것을 전부 확정한다.
            tail = text[len(self._confirmed):].strip()
            if tail:
                events.append(self._final_event(tail, text))
            self._reset_buffer()
            return events

        # LocalAgreement-2: 직전 결과와 접두사가 일치하는 만큼만 확정한다.
        agreed = _common_prefix(text, self._prev)
        self._prev = text

        if len(agreed) > len(self._confirmed):
            newly = agreed[len(self._confirmed):].strip()
            self._confirmed = agreed
            if newly:
                events.append(self._final_event(newly, text))

        rest = text[len(self._confirmed):].strip()
        if rest:
            events.append({"type": "partial", "text": rest})

        # 버퍼가 whisper 창(30초)에 가까워지면 확정된 앞부분을 잘라낸다.
        if self._buf.size > MAX_BUFFER_SECONDS * STT_SAMPLE_RATE:
            self._reset_buffer(keep_seconds=5.0)

        return events

    def _final_event(self, newly: str, whole: str) -> dict:
        start = self._offset
        end = self._offset + self._buf.size / STT_SAMPLE_RATE
        return {"type": "final", "text": newly, "start": round(start, 3), "end": round(end, 3)}

    def _reset_buffer(self, keep_seconds: float = 0.0) -> None:
        keep = int(keep_seconds * STT_SAMPLE_RATE)
        dropped = max(0, self._buf.size - keep)
        self._offset += dropped / STT_SAMPLE_RATE
        self._buf = self._buf[dropped:] if keep else np.zeros(0, dtype=np.float32)
        self._prev = ""
        self._confirmed = ""


def _new_session(*, language: str | None = None) -> _Session:
    return _Session(language=language)


def _install_stream_route(ctx):
    """/v1/audio/stream(WebSocket)을 이 엔진에만 붙인다.

    voiceapi.py 의 확장점(create_app(routes=...))을 쓴다 — "한 엔진에만 필요한 라우트는
    그 엔진이 직접 붙인다"는 규약대로, 공유 파일(voiceapi.py)은 건드리지 않는다.

    스트리밍은 요청/응답 한 쌍이 아니라 양방향이라 HTTP 인증 계약(Depends)으로 덮이지
    않는다. 그래서 인증을 여기서 따로 한다 — create_app 이 읽는 것과 같은 API_KEY
    환경변수를 그대로 읽어 쿼리스트링(?token=)이나 Authorization 헤더로 검사한다.
    브라우저 WebSocket API 는 커스텀 헤더를 못 붙이므로 쿼리스트링을 함께 받는다.

    규격:
      접속   ws://<host>:8104/v1/audio/stream?language=ko&token=<API_KEY>
      보내기 바이너리: PCM16LE mono 16kHz 원시 샘플 (헤더 없음)
             텍스트  : {"type":"eof"}        입력 종료, 최종 결과를 받는다
                      {"language":"ko"}      첫 프레임으로 설정 변경 (선택)
      받기   {"type":"partial","text":"..."}                    바뀔 수 있는 중간 결과
             {"type":"final","text":"...","start":..,"end":..}  확정 구간
             {"type":"done"}                                    처리 완료
             {"type":"error","detail":"..."}
    """
    app = ctx.app
    engine = ctx.engine
    api_key = os.getenv("API_KEY", "").strip() or None

    @app.websocket("/v1/audio/stream")
    async def audio_stream(ws: WebSocket) -> None:
        if api_key:
            token = ws.query_params.get("token", "")
            if not token:
                hdr = ws.headers.get("authorization", "")
                token = hdr[7:].strip() if hdr[:7].lower() == "bearer " else ""
            if token != api_key:
                await ws.close(code=1008, reason="Invalid API key")
                return

        await ws.accept()

        if not engine.ready:
            if engine.error:
                await ws.send_json({"type": "error", "detail": f"Engine initialization failed: {engine.error}"})
                await ws.close()
                return
            # 로딩 중이면 끊지 말고 기다린다. 스트리밍 클라이언트는 재접속 비용이 크다.
            await ws.send_json({"type": "loading"})
            await asyncio.get_running_loop().run_in_executor(None, engine.load)

        loop = asyncio.get_running_loop()
        language = ws.query_params.get("language") or None
        session = _new_session(language=language)

        async def emit(events: list[dict]) -> None:
            for ev in events or []:
                await ws.send_json(ev)

        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    return

                if (data := msg.get("bytes")) is not None:
                    if not data:
                        continue
                    # PCM16LE -> float32 (-1.0~1.0). voiceapi 의 STT 입력 규격과 같다.
                    pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
                    await emit(await loop.run_in_executor(None, session.feed, pcm))
                    continue

                text = msg.get("text")
                if text is None:
                    continue
                try:
                    payload = json.loads(text)
                except ValueError:
                    await ws.send_json({"type": "error", "detail": "Not JSON"})
                    continue

                if payload.get("type") == "eof":
                    await emit(await loop.run_in_executor(None, session.finish))
                    await ws.send_json({"type": "done"})
                    return
                if "language" in payload:
                    # 아직 오디오를 받기 전이면 세션을 다시 만든다.
                    language = payload["language"] or None
                    session = _new_session(language=language)

        except WebSocketDisconnect:
            return
        except Exception as exc:  # 엔진 쪽 예외를 클라이언트에 알려주고 닫는다
            log.exception("Streaming failed")
            try:
                await ws.send_json({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    return ["WS /v1/audio/stream"]


def transcribe(
    audio: np.ndarray,
    *,
    language: str | None,
    prompt: str | None,
    temperature: float,
    task: str,
) -> dict:
    """배치 요청도 받는다. 스트리밍 전용으로 두면 동작 확인이 번거롭다."""
    text, lang = _decode(audio[: int(30 * STT_SAMPLE_RATE)], language or DEFAULT_LANGUAGE)
    return {"text": text, "language": lang or "unknown", "segments": []}


spec = EngineSpec(
    name="whisper-streaming",
    kind="stt",
    model=MODEL,
    port=int(os.getenv("PORT", "8104")),
    languages=[],   # whisper 는 99개 언어 자동 감지
    sample_rate=STT_SAMPLE_RATE,
    extra={
        "backend": "transformers WhisperForConditionalGeneration (torch/ROCm)",
        "backend_note": "WhisperLive/whisper_streaming 은 faster-whisper(CTranslate2) 전제라 "
                        "ROCm 에서 GPU 를 못 쓴다. 배치 엔진(8103)과 같은 백엔드를 쓴다",
        "streaming": True,
        "policy": "LocalAgreement-2 (연속 두 추론의 공통 접두사까지만 확정)",
        "chunk_seconds": CHUNK_SECONDS,
        "device": DEVICE,
        "compute_type": DTYPE,
        "supports_translate": False,
        "license": "MIT (코드/모델 모두)",
    },
)

app = create_app(spec, loader=load, transcribe=transcribe, routes=_install_stream_route)

if __name__ == "__main__":
    run(app, spec)
