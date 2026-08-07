"""
voiceapi — 모든 STT/TTS 엔진 컨테이너가 공유하는 공통 HTTP 레이어.

새 엔진을 추가할 때 server.py 가 구현해야 하는 것은 두 가지뿐이다.

  STT: transcribe(audio, *, language, prompt, temperature, task) -> dict
       audio 는 16kHz mono float32 numpy 배열.
       반환 dict 키: text (필수), language, duration, segments

  TTS: synthesize(text, *, voice, language, speed) -> (wav, sample_rate)
       wav 는 mono float32 numpy 배열 (-1.0 ~ 1.0).
       voices() -> [{"id","name","language","gender"}, ...]

엔드포인트 규격, 오디오 디코딩/인코딩, API 키 인증, /health, /info 는
전부 여기서 처리하므로 엔진별로 다시 쓰지 않는다.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

log = logging.getLogger("voiceapi")

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

# STT 엔진 공통 입력 규격. 브라우저가 보내는 webm/opus 든 mp3 든 이 형태로 통일한다.
STT_SAMPLE_RATE = 16000

_MIME = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "flac": "audio/flac",
    "pcm": "application/octet-stream",
}


# ---------------------------------------------------------------- 오디오 변환


def decode_audio(raw: bytes, sample_rate: int = STT_SAMPLE_RATE) -> np.ndarray:
    """임의 포맷(wav/mp3/webm/ogg/m4a/flac)의 바이트를 mono float32 PCM 으로."""
    if not raw:
        raise HTTPException(400, "오디오 데이터가 비어 있습니다")
    cmd = [
        FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "s16le", "-ac", "1", "-ar", str(sample_rate),
        "pipe:1",
    ]
    p = subprocess.run(cmd, input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not p.stdout:
        detail = p.stderr.decode("utf-8", "replace").strip()[:300]
        raise HTTPException(400, f"오디오 디코딩 실패: {detail or 'ffmpeg 오류'}")
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _to_int16(wav: np.ndarray) -> np.ndarray:
    a = np.asarray(wav, dtype=np.float32).reshape(-1)
    return (np.clip(a, -1.0, 1.0) * 32767.0).astype(np.int16)


def _wav_bytes(pcm16: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


def encode_audio(wav: np.ndarray, sample_rate: int, fmt: str = "wav") -> tuple[bytes, str]:
    """float32 PCM 을 요청된 포맷으로. 반환값은 (바이트, content-type)."""
    fmt = (fmt or "wav").lower()
    if fmt not in _MIME:
        raise HTTPException(400, f"지원하지 않는 출력 형식: {fmt} (가능: {', '.join(_MIME)})")

    pcm16 = _to_int16(wav)
    if fmt == "pcm":
        return pcm16.tobytes(), _MIME[fmt]
    if fmt == "wav":
        return _wav_bytes(pcm16, sample_rate), _MIME[fmt]

    codec = {
        "mp3": ["-f", "mp3", "-b:a", "128k"],
        "opus": ["-f", "ogg", "-c:a", "libopus", "-b:a", "64k"],
        "flac": ["-f", "flac"],
    }[fmt]
    cmd = [
        FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "-i", "pipe:0",
        *codec, "pipe:1",
    ]
    p = subprocess.run(cmd, input=pcm16.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not p.stdout:
        detail = p.stderr.decode("utf-8", "replace").strip()[:300]
        raise HTTPException(500, f"오디오 인코딩 실패({fmt}): {detail or 'ffmpeg 오류'}")
    return p.stdout, _MIME[fmt]


# ---------------------------------------------------------------- 엔진 메타


@dataclass
class EngineSpec:
    name: str                                # 예: "whisper"
    kind: str                                # "stt" | "tts"
    model: str                               # 로드한 모델 식별자
    port: int
    languages: Sequence[str] = ()            # 빈 값이면 "다국어/자동"
    sample_rate: int = STT_SAMPLE_RATE       # TTS 는 출력 샘플레이트
    extra: dict[str, Any] = field(default_factory=dict)


class _Engine:
    """모델 로딩 상태를 들고 있는 래퍼. 로딩 중 요청은 503 으로 돌려보낸다."""

    def __init__(self, loader: Callable[[], Any] | None):
        self._loader = loader
        self.ready = loader is None
        self.error: str | None = None
        self.loaded_at: float | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self.ready or self._loader is None:
            return
        with self._lock:
            if self.ready:
                return
            t0 = time.time()
            try:
                self._loader()
            except Exception as exc:  # 모델 다운로드 실패 등
                self.error = f"{type(exc).__name__}: {exc}"
                log.exception("모델 로딩 실패")
                raise
            self.ready = True
            self.loaded_at = time.time()
            log.info("모델 로딩 완료 (%.1fs)", self.loaded_at - t0)

    def require(self) -> None:
        if self.error:
            raise HTTPException(503, f"엔진 초기화 실패: {self.error}")
        if not self.ready:
            raise HTTPException(503, "모델 로딩 중입니다. 잠시 후 다시 시도하세요")


# ---------------------------------------------------------------- 요청 스키마


class SpeechRequest(BaseModel):
    """OpenAI /v1/audio/speech 호환 + language 확장."""

    input: str = Field(..., description="합성할 텍스트")
    model: str | None = Field(None, description="호환용. 무시된다")
    voice: str | None = Field(None, description="보이스 id. 생략 시 엔진 기본값")
    language: str | None = Field(None, description="언어 코드. 생략 시 엔진 기본값")
    speed: float = Field(1.0, ge=0.25, le=4.0)
    response_format: str = Field("wav", description="wav|mp3|opus|flac|pcm")


# ---------------------------------------------------------------- 앱 팩토리


def _auth_dependency(api_key: str | None):
    async def check(request: Request) -> None:
        if not api_key:
            return
        hdr = request.headers.get("authorization", "")
        token = hdr[7:].strip() if hdr[:7].lower() == "bearer " else ""
        if not token:
            token = request.headers.get("x-api-key", "").strip()
        if token != api_key:
            raise HTTPException(401, "유효하지 않은 API 키")

    return check


def create_app(
    spec: EngineSpec,
    *,
    loader: Callable[[], Any] | None = None,
    transcribe: Callable[..., dict] | None = None,
    synthesize: Callable[..., tuple[np.ndarray, int]] | None = None,
    voices: Callable[[], list[dict]] | None = None,
) -> FastAPI:
    if spec.kind == "stt" and transcribe is None:
        raise ValueError("STT 엔진은 transcribe 를 넘겨야 합니다")
    if spec.kind == "tts" and synthesize is None:
        raise ValueError("TTS 엔진은 synthesize 를 넘겨야 합니다")

    api_key = os.getenv("API_KEY", "").strip() or None
    engine = _Engine(loader)
    started = time.time()

    app = FastAPI(
        title=f"{spec.name} ({spec.kind.upper()}) API",
        version="1.0.0",
        description=f"{spec.name} 엔진 단독 컨테이너. OpenAI Audio API 호환.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    auth = Depends(_auth_dependency(api_key))

    # ---- 수명주기 ----------------------------------------------------------

    @app.on_event("startup")
    async def _startup() -> None:
        # 기본은 기동 시 미리 로드. PRELOAD=0 이면 첫 요청까지 메모리를 아낀다.
        if os.getenv("PRELOAD", "1") not in ("0", "false", "no"):
            await asyncio.get_running_loop().run_in_executor(None, engine.load)

    # ---- 상태 --------------------------------------------------------------

    @app.get("/health", summary="헬스체크 (인증 불필요)")
    async def health() -> JSONResponse:
        body = {
            "status": "ok" if engine.ready else ("error" if engine.error else "loading"),
            "engine": spec.name,
            "kind": spec.kind,
            "model": spec.model,
            "ready": engine.ready,
            "uptime_s": round(time.time() - started, 1),
        }
        if engine.error:
            body["error"] = engine.error
        return JSONResponse(body, status_code=200 if engine.ready else 503)

    @app.get("/info", summary="엔진 정보", dependencies=[auth])
    async def info() -> dict:
        endpoints = (
            ["POST /v1/audio/transcriptions", "POST /v1/audio/translations", "POST /transcribe"]
            if spec.kind == "stt"
            else ["POST /v1/audio/speech", "POST /tts", "GET /v1/voices"]
        )
        return {
            "engine": spec.name,
            "kind": spec.kind,
            "model": spec.model,
            "port": spec.port,
            "languages": list(spec.languages) or ["auto"],
            "sample_rate": spec.sample_rate,
            "auth_required": bool(api_key),
            "endpoints": endpoints + ["GET /health", "GET /info", "GET /docs"],
            **spec.extra,
        }

    # ---- STT ---------------------------------------------------------------

    if spec.kind == "stt":

        async def _run_stt(
            file: UploadFile,
            language: str | None,
            prompt: str | None,
            temperature: float,
            response_format: str,
            task: str,
        ):
            engine.require()
            raw = await file.read()
            audio = decode_audio(raw, spec.sample_rate)
            if audio.size == 0:
                raise HTTPException(400, "디코딩 결과가 빈 오디오입니다")

            t0 = time.perf_counter()
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: transcribe(
                    audio,
                    language=language or None,
                    prompt=prompt or None,
                    temperature=temperature,
                    task=task,
                ),
            )
            elapsed = time.perf_counter() - t0

            text = (result.get("text") or "").strip()
            fmt = (response_format or "json").lower()
            if fmt == "text":
                return PlainTextResponse(text)
            body = {
                "text": text,
                "language": result.get("language"),
                "duration": round(audio.size / spec.sample_rate, 3),
                "engine": spec.name,
                "processing_s": round(elapsed, 3),
            }
            if fmt == "verbose_json":
                body["segments"] = result.get("segments", [])
            return JSONResponse(body)

        @app.post("/v1/audio/transcriptions", summary="음성 → 텍스트 (OpenAI 호환)", dependencies=[auth])
        async def transcriptions(
            file: UploadFile = File(..., description="오디오 파일"),
            model: str | None = Form(None),
            language: str | None = Form(None),
            prompt: str | None = Form(None),
            temperature: float = Form(0.0),
            response_format: str = Form("json"),
        ):
            return await _run_stt(file, language, prompt, temperature, response_format, "transcribe")

        @app.post("/v1/audio/translations", summary="음성 → 영어 텍스트 (지원 엔진만)", dependencies=[auth])
        async def translations(
            file: UploadFile = File(...),
            model: str | None = Form(None),
            prompt: str | None = Form(None),
            temperature: float = Form(0.0),
            response_format: str = Form("json"),
        ):
            if not spec.extra.get("supports_translate"):
                raise HTTPException(400, f"{spec.name} 은 음성→영어 번역을 지원하지 않습니다")
            return await _run_stt(file, None, prompt, temperature, response_format, "translate")

        @app.post("/transcribe", summary="/v1/audio/transcriptions 별칭", dependencies=[auth])
        async def transcribe_alias(
            file: UploadFile = File(...),
            language: str | None = Form(None),
            response_format: str = Form("json"),
        ):
            return await _run_stt(file, language, None, 0.0, response_format, "transcribe")

    # ---- TTS ---------------------------------------------------------------

    if spec.kind == "tts":

        @app.get("/v1/voices", summary="사용 가능한 보이스 목록", dependencies=[auth])
        async def list_voices() -> dict:
            engine.require()
            return {"engine": spec.name, "voices": voices() if voices else []}

        async def _run_tts(req: SpeechRequest):
            engine.require()
            text = (req.input or "").strip()
            if not text:
                raise HTTPException(400, "input 이 비어 있습니다")
            if len(text) > int(os.getenv("MAX_CHARS", "5000")):
                raise HTTPException(413, "input 이 너무 깁니다")

            t0 = time.perf_counter()
            wav, sr = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: synthesize(
                    text,
                    voice=req.voice or None,
                    language=req.language or None,
                    speed=req.speed,
                ),
            )
            elapsed = time.perf_counter() - t0
            data, mime = encode_audio(wav, sr, req.response_format)
            return Response(
                content=data,
                media_type=mime,
                headers={
                    "X-Engine": spec.name,
                    "X-Sample-Rate": str(sr),
                    "X-Audio-Duration": f"{np.asarray(wav).reshape(-1).size / sr:.3f}",
                    "X-Processing-Seconds": f"{elapsed:.3f}",
                },
            )

        @app.post("/v1/audio/speech", summary="텍스트 → 음성 (OpenAI 호환)", dependencies=[auth])
        async def speech(req: SpeechRequest):
            return await _run_tts(req)

        @app.post("/tts", summary="/v1/audio/speech 별칭", dependencies=[auth])
        async def tts_alias(req: SpeechRequest):
            return await _run_tts(req)

    return app


def run(app: FastAPI, spec: EngineSpec) -> None:
    """컨테이너 엔트리포인트. PORT 는 환경변수로 덮어쓸 수 있다."""
    import uvicorn

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", spec.port)),
        workers=1,  # 모델을 메모리에 한 번만 올린다
        timeout_keep_alive=int(os.getenv("KEEPALIVE", "30")),
        access_log=os.getenv("ACCESS_LOG", "0") not in ("0", "false", "no"),
    )
