"""
voiceapi — 모든 음성 엔진 컨테이너가 공유하는 공통 HTTP 레이어.

엔진의 종류(kind)마다 붙는 라우트가 다르다. 종류는 **레지스트리**에 등록되어 있고
create_app 은 spec.kind 이름으로 골라 쓴다 (소스에 종류를 분기문으로 박지 않는다).
새 종류를 추가하려면 설치 함수를 만들어 register_kind() 로 등록하면 되고,
한 엔진에만 필요한 라우트는 create_app(routes=...) 로 그 엔진이 직접 붙인다.

server.py 가 구현해야 하는 것은 자기 종류에 해당하는 함수뿐이다.

  stt:     transcribe(audio, *, language, prompt, temperature, task) -> dict
           audio 는 16kHz mono float32 numpy 배열.
           반환 dict 키: text (필수), language, duration, segments

  tts:     synthesize(text, *, voice, language, speed) -> (wav, sample_rate)
           wav 는 mono float32 numpy 배열 (-1.0 ~ 1.0).
           voices() -> [{"id","name","language","gender"}, ...]

  speaker: embed(audio) -> np.ndarray
           audio 는 16kHz mono float32 numpy 배열.
           반환은 고정 차원 화자 임베딩(L2 정규화 권장) 1차원 배열.

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
    kind: str                                # 등록된 종류: "stt" | "tts" | "speaker"
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


# ---------------------------------------------------------------- 라우트 설치 컨텍스트


@dataclass
class EngineContext:
    """라우트 설치 함수에 넘기는 것들. 종류별 설치 함수와 엔진 전용 routes= 가 같이 쓴다."""

    app: FastAPI
    spec: EngineSpec
    auth: Any                                # 라우트에 dependencies=[ctx.auth] 로 건다
    engine: _Engine
    handlers: dict[str, Any]                 # transcribe / synthesize / voices / embed ...

    def require(self) -> None:
        """모델이 준비됐는지 확인. 로딩 중이면 503."""
        self.engine.require()

    def handler(self, name: str) -> Any:
        fn = self.handlers.get(name)
        if fn is None:
            raise ValueError(f"{self.spec.kind} 엔진은 {name}() 를 넘겨야 합니다")
        return fn

    async def offload(self, fn: Callable[[], Any]) -> Any:
        """추론은 이벤트 루프를 막지 않도록 스레드로 넘긴다."""
        return await asyncio.get_running_loop().run_in_executor(None, fn)


# 종류 → 라우트 설치 함수. 설치 함수는 /info 에 실을 엔드포인트 목록을 돌려준다.
KIND_ROUTES: dict[str, Callable[[EngineContext], list[str]]] = {}


def register_kind(kind: str, installer: Callable[[EngineContext], list[str]]) -> None:
    """엔진 종류를 등록한다. 같은 이름을 다시 등록하면 덮어쓴다."""
    KIND_ROUTES[kind] = installer


# ---------------------------------------------------------------- 종류별 라우트: STT


def _install_stt(ctx: EngineContext) -> list[str]:
    app, spec = ctx.app, ctx.spec
    transcribe = ctx.handler("transcribe")
    auth = ctx.auth

    async def _run_stt(
        file: UploadFile,
        language: str | None,
        prompt: str | None,
        temperature: float,
        response_format: str,
        task: str,
    ):
        ctx.require()
        raw = await file.read()
        audio = decode_audio(raw, spec.sample_rate)
        if audio.size == 0:
            raise HTTPException(400, "디코딩 결과가 빈 오디오입니다")

        t0 = time.perf_counter()
        result = await ctx.offload(
            lambda: transcribe(
                audio,
                language=language or None,
                prompt=prompt or None,
                temperature=temperature,
                task=task,
            )
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

    return ["POST /v1/audio/transcriptions", "POST /v1/audio/translations", "POST /transcribe"]


# ---------------------------------------------------------------- 종류별 라우트: TTS


def _install_tts(ctx: EngineContext) -> list[str]:
    app, spec = ctx.app, ctx.spec
    synthesize = ctx.handler("synthesize")
    voices = ctx.handlers.get("voices")
    auth = ctx.auth

    @app.get("/v1/voices", summary="사용 가능한 보이스 목록", dependencies=[auth])
    async def list_voices() -> dict:
        ctx.require()
        return {"engine": spec.name, "voices": voices() if voices else []}

    async def _run_tts(req: SpeechRequest):
        ctx.require()
        text = (req.input or "").strip()
        if not text:
            raise HTTPException(400, "input 이 비어 있습니다")
        if len(text) > int(os.getenv("MAX_CHARS", "5000")):
            raise HTTPException(413, "input 이 너무 깁니다")

        t0 = time.perf_counter()
        wav, sr = await ctx.offload(
            lambda: synthesize(
                text,
                voice=req.voice or None,
                language=req.language or None,
                speed=req.speed,
            )
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

    return ["POST /v1/audio/speech", "POST /tts", "GET /v1/voices"]


# ---------------------------------------------------------------- 종류별 라우트: SPEAKER


def _install_speaker(ctx: EngineContext) -> list[str]:
    """화자 임베딩. 오디오를 받아 고정 차원 벡터를 돌려준다.

    임베딩끼리의 비교(등록 화자 매칭)는 호출하는 쪽이 한다. 여기서는 벡터 생성과,
    등록용으로 여러 발화를 평균 내는 것, 두 오디오를 바로 비교하는 것까지만 맡는다.
    """

    app, spec = ctx.app, ctx.spec
    embed = ctx.handler("embed")
    auth = ctx.auth

    # 너무 짧은 오디오는 임베딩이 불안정하고, 너무 긴 것은 메모리를 먹는다.
    min_s = float(os.getenv("MIN_AUDIO_SECONDS", "1.0"))
    max_s = float(os.getenv("MAX_AUDIO_SECONDS", "60"))
    # /info 가 같은 값을 보고하도록 여기서 심는다 (설정을 두 군데서 읽지 않는다).
    spec.extra.setdefault("min_audio_seconds", min_s)
    spec.extra.setdefault("max_audio_seconds", max_s)

    async def _decode(file: UploadFile) -> tuple[np.ndarray, float]:
        raw = await file.read()
        audio = decode_audio(raw, spec.sample_rate)
        seconds = audio.size / spec.sample_rate
        name = file.filename or "audio"
        if audio.size == 0:
            raise HTTPException(400, f"디코딩 결과가 빈 오디오입니다: {name}")
        if seconds < min_s:
            raise HTTPException(
                400,
                f"오디오가 너무 짧습니다: {name} {seconds:.2f}초 "
                f"(최소 {min_s:.2f}초). 짧은 발화는 화자 임베딩이 불안정합니다",
            )
        if seconds > max_s:
            raise HTTPException(
                413,
                f"오디오가 너무 깁니다: {name} {seconds:.1f}초 (최대 {max_s:.0f}초)",
            )
        return audio, seconds

    async def _embed_one(file: UploadFile) -> tuple[np.ndarray, float]:
        audio, seconds = await _decode(file)
        vec = await ctx.offload(lambda: embed(audio))
        return np.asarray(vec, dtype=np.float32).reshape(-1), seconds

    def _unit(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    @app.post("/v1/speaker/embed", summary="음성 → 화자 임베딩", dependencies=[auth])
    async def speaker_embed(file: UploadFile = File(..., description="오디오 파일")):
        ctx.require()
        t0 = time.perf_counter()
        vec, seconds = await _embed_one(file)
        elapsed = time.perf_counter() - t0
        return JSONResponse(
            {
                "engine": spec.name,
                "model": spec.model,
                "dim": int(vec.size),
                "embedding": [float(x) for x in vec],
                "duration": round(seconds, 3),
                "processing_s": round(elapsed, 3),
            }
        )

    @app.post(
        "/v1/speaker/enroll",
        summary="여러 발화 → 평균 화자 임베딩 (등록용)",
        dependencies=[auth],
    )
    async def speaker_enroll(files: list[UploadFile] = File(..., description="오디오 파일 여러 개")):
        ctx.require()
        if not files:
            raise HTTPException(400, "files 가 비어 있습니다")
        t0 = time.perf_counter()
        vecs: list[np.ndarray] = []
        per_file: list[dict] = []
        for f in files:
            vec, seconds = await _embed_one(f)
            vecs.append(vec)
            per_file.append({"filename": f.filename, "duration": round(seconds, 3)})

        stack = np.stack(vecs)
        mean = _unit(stack.mean(axis=0))
        sims = stack @ stack.T
        for i, item in enumerate(per_file):
            item["similarity_to_mean"] = round(float(stack[i] @ mean), 4)
        # 등록 발화 중 하나가 다른 화자면 최솟값이 눈에 띄게 낮아진다.
        off = sims[~np.eye(len(vecs), dtype=bool)] if len(vecs) > 1 else np.array([1.0])
        elapsed = time.perf_counter() - t0
        return JSONResponse(
            {
                "engine": spec.name,
                "model": spec.model,
                "dim": int(mean.size),
                "embedding": [float(x) for x in mean],
                "count": len(vecs),
                "duration": round(sum(i["duration"] for i in per_file), 3),
                "files": per_file,
                "min_pairwise_similarity": round(float(off.min()), 4),
                "processing_s": round(elapsed, 3),
            }
        )

    @app.post("/v1/speaker/compare", summary="두 오디오의 화자 유사도", dependencies=[auth])
    async def speaker_compare(
        file_a: UploadFile = File(..., description="오디오 A"),
        file_b: UploadFile = File(..., description="오디오 B"),
        threshold: float | None = Form(None, description="생략 시 엔진 권장 임계값"),
    ):
        ctx.require()
        t0 = time.perf_counter()
        va, sa = await _embed_one(file_a)
        vb, sb = await _embed_one(file_b)
        sim = float(_unit(va) @ _unit(vb))
        thr = threshold if threshold is not None else spec.extra.get("similarity_threshold")
        elapsed = time.perf_counter() - t0
        return JSONResponse(
            {
                "engine": spec.name,
                "similarity": round(sim, 4),
                "threshold": thr,
                "same_speaker": None if thr is None else bool(sim >= float(thr)),
                "duration": [round(sa, 3), round(sb, 3)],
                "processing_s": round(elapsed, 3),
            }
        )

    return [
        "POST /v1/speaker/embed",
        "POST /v1/speaker/enroll",
        "POST /v1/speaker/compare",
    ]


register_kind("stt", _install_stt)
register_kind("tts", _install_tts)
register_kind("speaker", _install_speaker)


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
    embed: Callable[..., np.ndarray] | None = None,
    routes: Callable[[EngineContext], Sequence[str] | None] | None = None,
) -> FastAPI:
    if spec.kind not in KIND_ROUTES:
        raise ValueError(
            f"등록되지 않은 엔진 종류: {spec.kind} (등록된 것: {', '.join(sorted(KIND_ROUTES))})"
        )

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

    # ---- 종류별 라우트 ------------------------------------------------------

    ctx = EngineContext(
        app=app,
        spec=spec,
        auth=auth,
        engine=engine,
        handlers={
            "transcribe": transcribe,
            "synthesize": synthesize,
            "voices": voices,
            "embed": embed,
        },
    )
    endpoints = list(KIND_ROUTES[spec.kind](ctx))
    if routes is not None:
        endpoints += list(routes(ctx) or [])

    @app.get("/info", summary="엔진 정보", dependencies=[auth])
    async def info() -> dict:
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
