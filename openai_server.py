#!/usr/bin/env python3
"""
OpenAI-compatible TTS API server for faster-qwen3-tts.

Exposes:
  POST /v1/audio/speech          – OpenAI-compatible one-shot synthesis
  POST /v1/tts/sessions          – create a streaming session (returns GET URL)
  GET  /v1/tts/sessions/{id}/stream – progressive WebM/Opus (or WAV) audio stream

Usage:
    pip install "faster-qwen3-tts[demo]"

    # Single default voice:
    python examples/openai_server.py \\
        --ref-audio voice.wav --ref-text "Reference transcription" \\
        --language English

    # Multiple named voices from a JSON config:
    python examples/openai_server.py --voices voices.json

    # Custom model and port:
    python examples/openai_server.py \\
        --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
        --ref-audio voice.wav --ref-text "transcript" \\
        --port 8000

Voices config (voices.json):
    {
        "alloy": {"ref_audio": "voice.wav", "ref_text": "...", "language": "English"},
        "echo":  {"ref_audio": "voice2.wav", "ref_text": "...", "language": "English"}
    }

API usage:
    # Direct synthesis (WebM/Opus):
    curl -s http://localhost:8000/v1/audio/speech \\
        -H "Content-Type: application/json" \\
        -d '{"model":"tts-1","input":"Hello!","voice":"alloy","response_format":"webm"}' \\
        --output speech.webm

    # Session-based streaming:
    SESSION=$(curl -s http://localhost:8000/v1/tts/sessions \\
        -H "Content-Type: application/json" \\
        -d '{"text":"Long text here...","voice":"alloy","format":"webm"}' | jq -r .stream_url)
    curl -s "http://localhost:8000${SESSION}" --output speech.webm
"""
import argparse
import asyncio
import contextlib
import io
import json
import logging
import os
import queue
import re
import struct
import sys
import threading
import time
import uuid
from typing import AsyncGenerator, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app = FastAPI(title="faster-qwen3-tts OpenAI-compatible API")

tts_model = None
voices: dict = {}
default_voice: Optional[str] = None
SAMPLE_RATE = 24000  # updated once the model loads
_model_lock = threading.Lock()  # prevent concurrent GPU inference
MAX_LOG_TEXT_CHARS = 1000

# Session store: session_id -> SessionEntry
_sessions: dict[str, "SessionEntry"] = {}
_SESSION_TTL_AFTER_STREAM = 30  # seconds to keep session after stream completes
_SESSION_TTL_UNUSED = 300       # 5 minutes if never streamed


class SessionEntry:
    """In-memory TTS session created by POST /v1/tts/sessions."""
    __slots__ = (
        "session_id", "text", "voice", "fmt", "speed",
        "created_at", "streamed", "finished_at",
    )

    def __init__(self, session_id: str, text: str, voice: str, fmt: str, speed: float):
        self.session_id = session_id
        self.text = text
        self.voice = voice
        self.fmt = fmt
        self.speed = speed
        self.created_at = time.monotonic()
        self.streamed = False
        self.finished_at: Optional[float] = None

    def is_expired(self) -> bool:
        now = time.monotonic()
        if self.finished_at is not None:
            return (now - self.finished_at) > _SESSION_TTL_AFTER_STREAM
        return (now - self.created_at) > _SESSION_TTL_UNUSED


def _preview_text(text: str, max_chars: int = MAX_LOG_TEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...(+{len(text) - max_chars} chars)"


def _log_tts(event: str, **kwargs):
    logger.info("[tts-service] %s", json.dumps({"event": event, **kwargs}, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Text segmentation (server-side batching)
# ---------------------------------------------------------------------------

_SEGMENT_MIN_CHARS = 80
_SEGMENT_MAX_CHARS = 400


def segment_text(text: str) -> list[str]:
    """Split text into synthesis-friendly segments.

    Strategy (per user preference):
      1. Split on double newlines first.
      2. If any resulting block is still long, split on single newlines.
      3. Merge short fragments forward so each segment >= _SEGMENT_MIN_CHARS.
      4. Cap segments at _SEGMENT_MAX_CHARS by splitting on sentence boundaries.
    """
    text = text.strip()
    if not text:
        return []

    # Phase 1: split on double newlines
    blocks = [b.strip() for b in re.split(r"\n\n+", text) if b.strip()]

    # Phase 2: split remaining large blocks on single newlines
    expanded: list[str] = []
    for block in blocks:
        if len(block) > _SEGMENT_MAX_CHARS:
            sub_blocks = [s.strip() for s in block.split("\n") if s.strip()]
            expanded.extend(sub_blocks)
        else:
            expanded.append(block)

    # Phase 3: further split any block still over max on sentence boundaries
    split_further: list[str] = []
    for block in expanded:
        if len(block) <= _SEGMENT_MAX_CHARS:
            split_further.append(block)
        else:
            # Split on sentence-ending punctuation followed by space
            sentences = re.split(r"(?<=[.!?…])\s+", block)
            split_further.extend(s.strip() for s in sentences if s.strip())

    # Phase 4: merge short fragments forward
    merged: list[str] = []
    for segment in split_further:
        if merged and len(merged[-1]) < _SEGMENT_MIN_CHARS:
            merged[-1] = f"{merged[-1]} {segment}"
        else:
            merged.append(segment)

    return merged if merged else [text]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: str = "wav"  # wav | pcm | mp3 | webm
    speed: float = 1.0           # accepted but not yet applied


class SessionCreateRequest(BaseModel):
    text: str
    voice: str = "default"
    format: str = Field(default="webm", description="webm | wav | pcm")
    speed: float = 1.0


class SessionCreateResponse(BaseModel):
    session_id: str
    stream_url: str
    segments: int
    text_length: int


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _to_pcm16(pcm: np.ndarray) -> bytes:
    """Convert float32 numpy array to raw 16-bit little-endian PCM bytes."""
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def _to_pcm_f32le(pcm: np.ndarray) -> bytes:
    """Convert float32 numpy array to raw float32 little-endian bytes for ffmpeg."""
    return np.asarray(pcm, dtype=np.float32).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    """Build a WAV header.  Use data_len=0xFFFFFFFF for streaming (unknown size)."""
    n_channels = 1
    bits = 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate,
                          byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to a complete WAV file in memory."""
    raw = _to_pcm16(pcm)
    return _wav_header(sample_rate, len(raw)) + raw


def _to_mp3_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to MP3 bytes (requires pydub + ffmpeg)."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="response_format='mp3' requires pydub: pip install pydub",
        )
    segment = AudioSegment(
        _to_pcm16(pcm),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    buf = io.BytesIO()
    segment.export(buf, format="mp3")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# WebM/Opus streaming encoder via ffmpeg
# ---------------------------------------------------------------------------


async def _pcm_to_webm_opus(
    pcm_source: AsyncGenerator[bytes, None],
    sample_rate: int = 24000,
    bitrate: str = "24k",
) -> AsyncGenerator[bytes, None]:
    """Pipe raw float32 PCM through ffmpeg libopus encoder -> WebM/Opus stream.

    Yields WebM bytes as they become available.  The ffmpeg process is started
    once and fed all PCM from *pcm_source*; output is read in 4 KB chunks to
    keep backpressure reasonable.
    """
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-f", "f32le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-i", "pipe:0",
        "-c:a", "libopus",
        "-application", "voip",
        "-b:a", bitrate,
        "-vbr", "on",
        "-compression_level", "10",
        "-frame_duration", "20",
        "-f", "webm",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    writer_error: Optional[Exception] = None

    async def _writer():
        nonlocal writer_error
        try:
            async for pcm_bytes in pcm_source:
                if proc.stdin is None:
                    break
                proc.stdin.write(pcm_bytes)
                await proc.stdin.drain()
        except Exception as exc:
            writer_error = exc
        finally:
            if proc.stdin is not None:
                proc.stdin.close()
                await proc.stdin.wait_closed()

    writer_task = asyncio.create_task(_writer())

    try:
        assert proc.stdout is not None
        while True:
            data = await proc.stdout.read(4096)
            if not data:
                break
            yield data
    finally:
        await writer_task
        await proc.wait()
        if writer_error is not None:
            raise writer_error


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------


def resolve_voice(voice_name: str) -> dict:
    """Return voice config dict or fall back to default, else raise 400."""
    if voice_name in voices:
        return voices[voice_name]
    if default_voice and default_voice in voices:
        logger.warning(
            "Voice %r not configured; falling back to default voice %r",
            voice_name,
            default_voice,
        )
        return voices[default_voice]
    raise HTTPException(
        status_code=400,
        detail=(
            f"Voice {voice_name!r} is not configured. "
            f"Available voices: {list(voices.keys())}"
        ),
    )


# ---------------------------------------------------------------------------
# Streaming helper: run sync generator in a background thread
# ---------------------------------------------------------------------------


async def _stream_chunks(
    voice_cfg: dict,
    text: str,
    request_id: str,
) -> AsyncGenerator[tuple[bytes, dict], None]:
    """
    Run generate_voice_clone_streaming in a background thread and yield
    raw PCM bytes (int16) for each chunk as they arrive.
    """
    q: queue.Queue = queue.Queue()
    _DONE = object()

    def producer():
        try:
            with _model_lock:
                for chunk, _sr, timing in tts_model.generate_voice_clone_streaming(
                    text=text,
                    language=voice_cfg.get("language", "Auto"),
                    ref_audio=voice_cfg["ref_audio"],
                    ref_text=voice_cfg.get("ref_text", ""),
                    chunk_size=voice_cfg.get("chunk_size", 12),
                    non_streaming_mode=False,
                ):
                    q.put((chunk, timing))
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(_DONE)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _DONE:
            break
        if isinstance(item, Exception):
            raise item
        chunk, timing = item
        yield _to_pcm16(chunk), timing


async def _stream_chunks_f32(
    voice_cfg: dict,
    text: str,
    request_id: str,
) -> AsyncGenerator[tuple[bytes, dict], None]:
    """Like _stream_chunks but yields raw float32 bytes for the opus encoder."""
    q: queue.Queue = queue.Queue()
    _DONE = object()

    def producer():
        try:
            with _model_lock:
                for chunk, _sr, timing in tts_model.generate_voice_clone_streaming(
                    text=text,
                    language=voice_cfg.get("language", "Auto"),
                    ref_audio=voice_cfg["ref_audio"],
                    ref_text=voice_cfg.get("ref_text", ""),
                    chunk_size=voice_cfg.get("chunk_size", 12),
                    non_streaming_mode=False,
                ):
                    q.put((chunk, timing))
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(_DONE)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _DONE:
            break
        if isinstance(item, Exception):
            raise item
        chunk, timing = item
        yield _to_pcm_f32le(chunk), timing


async def _multi_segment_pcm_f32(
    voice_cfg: dict,
    segments: list[str],
    request_id: str,
) -> AsyncGenerator[bytes, None]:
    """Synthesize multiple text segments sequentially, yielding raw f32le PCM.

    This is the core pipeline for session-based streaming: the server owns
    segmentation and produces one continuous PCM stream across all segments.
    """
    for idx, seg in enumerate(segments):
        seg_start = time.perf_counter()
        chunk_count = 0
        _log_tts(
            "segment_start",
            request_id=request_id,
            segment_index=idx,
            segment_count=len(segments),
            segment_length=len(seg),
            segment_text=_preview_text(seg, 200),
        )
        async for pcm_bytes, timing in _stream_chunks_f32(voice_cfg, seg, request_id):
            chunk_count += 1
            yield pcm_bytes
        _log_tts(
            "segment_complete",
            request_id=request_id,
            segment_index=idx,
            chunks=chunk_count,
            elapsed_ms=round((time.perf_counter() - seg_start) * 1000, 2),
        )


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


def _gc_sessions():
    """Remove expired sessions.  Called lazily on session creation."""
    expired = [sid for sid, s in _sessions.items() if s.is_expired()]
    for sid in expired:
        del _sessions[sid]
    if expired:
        _log_tts("sessions_gc", removed=len(expired), remaining=len(_sessions))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": tts_model is not None}


@app.get("/v1/voices")
async def list_voices():
    return {
        "voices": [
            {"name": name, "language": cfg.get("language", "Auto")}
            for name, cfg in voices.items()
        ]
    }


# ---- Direct synthesis (backward-compatible OpenAI endpoint) ----


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest, request: Request):
    request_id = request.headers.get("x-request-id") or f"tts-{int(time.time() * 1000)}"
    started_at = time.perf_counter()

    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' text is empty")

    voice_cfg = resolve_voice(req.voice)
    fmt = req.response_format.lower()
    chunk_size = int(voice_cfg.get("chunk_size", 12))
    model_mode = voice_cfg.get("mode", "icl")

    _log_tts(
        "request_received",
        request_id=request_id,
        model=req.model,
        voice=req.voice,
        response_format=fmt,
        speed=req.speed,
        input_length=len(req.input),
        input_text=_preview_text(req.input),
        selected_language=voice_cfg.get("language", "Auto"),
        selected_ref_audio=voice_cfg.get("ref_audio"),
        selected_ref_text=voice_cfg.get("ref_text", ""),
        selected_chunk_size=chunk_size,
        selected_mode=model_mode,
    )

    _CONTENT_TYPES = {
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "mp3": "audio/mpeg",
        "webm": 'audio/webm; codecs="opus"',
    }
    if fmt not in _CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"response_format {fmt!r} not supported. Use: wav, pcm, mp3, webm",
        )
    content_type = _CONTENT_TYPES[fmt]

    # --- MP3: generate all audio, then encode (non-streaming) ---
    if fmt == "mp3":
        loop = asyncio.get_event_loop()

        def _generate():
            with _model_lock:
                return tts_model.generate_voice_clone(
                    text=req.input,
                    language=voice_cfg.get("language", "Auto"),
                    ref_audio=voice_cfg["ref_audio"],
                    ref_text=voice_cfg.get("ref_text", ""),
                )

        audio_arrays, sr = await loop.run_in_executor(None, _generate)
        audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)
        _log_tts(
            "request_complete_mp3",
            request_id=request_id,
            sample_rate=sr,
            audio_samples=len(audio),
            elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )
        return Response(content=_to_mp3_bytes(audio, sr), media_type=content_type)

    # --- WebM/Opus: stream through ffmpeg encoder ---
    if fmt == "webm":
        segments = segment_text(req.input)
        _log_tts(
            "webm_stream_start",
            request_id=request_id,
            segments=len(segments),
            segment_lengths=[len(s) for s in segments],
        )

        pcm_gen = _multi_segment_pcm_f32(voice_cfg, segments, request_id)

        async def webm_stream():
            total_bytes = 0
            chunk_count = 0
            first_chunk_at = None
            try:
                async for webm_bytes in _pcm_to_webm_opus(pcm_gen, SAMPLE_RATE):
                    chunk_count += 1
                    total_bytes += len(webm_bytes)
                    if first_chunk_at is None:
                        first_chunk_at = time.perf_counter()
                        _log_tts(
                            "stream_ttfa",
                            request_id=request_id,
                            ttfa_ms=round((first_chunk_at - started_at) * 1000, 2),
                        )
                    yield webm_bytes
            except Exception as exc:
                _log_tts(
                    "stream_error",
                    request_id=request_id,
                    error=str(exc),
                    elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
                )
                raise
            finally:
                _log_tts(
                    "request_complete_webm",
                    request_id=request_id,
                    chunk_count=chunk_count,
                    total_bytes=total_bytes,
                    elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
                )

        response = StreamingResponse(webm_stream(), media_type=content_type)
        response.headers["X-Request-Id"] = request_id
        return response

    # --- WAV / PCM: stream chunks as they are generated ---
    async def audio_stream():
        total_bytes = 0
        chunk_count = 0
        first_chunk_at = None
        if fmt == "wav":
            header = _wav_header(SAMPLE_RATE)
            total_bytes += len(header)
            yield header  # stream with unknown data length
        try:
            async for raw_chunk, timing in _stream_chunks(voice_cfg, req.input, request_id):
                chunk_count += 1
                total_bytes += len(raw_chunk)
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                    _log_tts(
                        "stream_ttfa",
                        request_id=request_id,
                        ttfa_ms=round((first_chunk_at - started_at) * 1000, 2),
                    )
                _log_tts(
                    "stream_chunk",
                    request_id=request_id,
                    chunk_index=chunk_count,
                    chunk_bytes=len(raw_chunk),
                    elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
                    model_timing=timing,
                )
                yield raw_chunk
        except Exception as exc:
            _log_tts(
                "stream_error",
                request_id=request_id,
                error=str(exc),
                elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )
            raise
        finally:
            _log_tts(
                "request_complete_stream",
                request_id=request_id,
                chunk_count=chunk_count,
                total_bytes=total_bytes,
                elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )

    response = StreamingResponse(audio_stream(), media_type=content_type)
    response.headers["X-Request-Id"] = request_id
    return response


# ---- Session-based streaming endpoints ----


@app.post("/v1/tts/sessions", response_model=SessionCreateResponse)
async def create_session(req: SessionCreateRequest, request: Request):
    """Create a TTS session.  Returns a stream URL the client can point
    an <audio> element or ffplay at."""
    request_id = request.headers.get("x-request-id") or f"tts-{int(time.time() * 1000)}"

    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    fmt = req.format.lower()
    if fmt not in ("webm", "wav", "pcm"):
        raise HTTPException(status_code=400, detail=f"format {fmt!r} not supported for sessions. Use: webm, wav, pcm")

    # Validate voice eagerly
    resolve_voice(req.voice)

    _gc_sessions()

    session_id = str(uuid.uuid4())
    entry = SessionEntry(
        session_id=session_id,
        text=text,
        voice=req.voice,
        fmt=fmt,
        speed=req.speed,
    )
    _sessions[session_id] = entry

    segments = segment_text(text)

    _log_tts(
        "session_created",
        request_id=request_id,
        session_id=session_id,
        voice=req.voice,
        format=fmt,
        text_length=len(text),
        segments=len(segments),
        segment_lengths=[len(s) for s in segments],
    )

    return SessionCreateResponse(
        session_id=session_id,
        stream_url=f"/v1/tts/sessions/{session_id}/stream",
        segments=len(segments),
        text_length=len(text),
    )


@app.get("/v1/tts/sessions/{session_id}/stream")
async def stream_session(session_id: str, request: Request):
    """Stream audio for a previously created session.  Consume-once."""
    entry = _sessions.get(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if entry.streamed:
        raise HTTPException(status_code=409, detail="Session already streamed (consume-once)")
    if entry.is_expired():
        _sessions.pop(session_id, None)
        raise HTTPException(status_code=410, detail="Session expired")

    entry.streamed = True
    request_id = f"sess-{session_id[:8]}"
    started_at = time.perf_counter()

    voice_cfg = resolve_voice(entry.voice)
    segments = segment_text(entry.text)
    fmt = entry.fmt

    _CONTENT_TYPES = {
        "webm": 'audio/webm; codecs="opus"',
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    content_type = _CONTENT_TYPES[fmt]

    _log_tts(
        "session_stream_start",
        request_id=request_id,
        session_id=session_id,
        format=fmt,
        segments=len(segments),
        text_length=len(entry.text),
    )

    if fmt == "webm":
        pcm_gen = _multi_segment_pcm_f32(voice_cfg, segments, request_id)

        async def webm_gen():
            total_bytes = 0
            chunk_count = 0
            first_at = None
            try:
                async for data in _pcm_to_webm_opus(pcm_gen, SAMPLE_RATE):
                    chunk_count += 1
                    total_bytes += len(data)
                    if first_at is None:
                        first_at = time.perf_counter()
                        _log_tts("stream_ttfa", request_id=request_id, ttfa_ms=round((first_at - started_at) * 1000, 2))
                    yield data
            except Exception as exc:
                _log_tts("stream_error", request_id=request_id, error=str(exc))
                raise
            finally:
                entry.finished_at = time.monotonic()
                _log_tts(
                    "session_stream_complete",
                    request_id=request_id,
                    session_id=session_id,
                    chunk_count=chunk_count,
                    total_bytes=total_bytes,
                    elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
                )

        resp = StreamingResponse(webm_gen(), media_type=content_type)
        resp.headers["X-Request-Id"] = request_id
        resp.headers["X-Session-Id"] = session_id
        return resp

    # WAV / PCM path (for TUI or fallback)
    async def raw_gen():
        total_bytes = 0
        chunk_count = 0
        first_at = None
        if fmt == "wav":
            header = _wav_header(SAMPLE_RATE)
            total_bytes += len(header)
            yield header
        try:
            for seg in segments:
                async for raw_chunk, _timing in _stream_chunks(voice_cfg, seg, request_id):
                    chunk_count += 1
                    total_bytes += len(raw_chunk)
                    if first_at is None:
                        first_at = time.perf_counter()
                        _log_tts("stream_ttfa", request_id=request_id, ttfa_ms=round((first_at - started_at) * 1000, 2))
                    yield raw_chunk
        except Exception as exc:
            _log_tts("stream_error", request_id=request_id, error=str(exc))
            raise
        finally:
            entry.finished_at = time.monotonic()
            _log_tts(
                "session_stream_complete",
                request_id=request_id,
                session_id=session_id,
                chunk_count=chunk_count,
                total_bytes=total_bytes,
                elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )

    resp = StreamingResponse(raw_gen(), media_type=content_type)
    resp.headers["X-Request-Id"] = request_id
    resp.headers["X-Session-Id"] = session_id
    return resp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="OpenAI-compatible TTS server for faster-qwen3-tts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        default=os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        help="HuggingFace model ID or local path (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)",
    )
    p.add_argument(
        "--voices",
        default=os.environ.get("QWEN_TTS_VOICES"),
        metavar="FILE",
        help="JSON file mapping voice names to {ref_audio, ref_text, language}",
    )
    p.add_argument(
        "--ref-audio",
        default=os.environ.get("QWEN_TTS_REF_AUDIO"),
        metavar="FILE",
        help="Reference audio file when --voices is not used",
    )
    p.add_argument(
        "--ref-text",
        default=os.environ.get("QWEN_TTS_REF_TEXT", ""),
        help="Transcript of --ref-audio",
    )
    p.add_argument(
        "--language",
        default=os.environ.get("QWEN_TTS_LANGUAGE", "Auto"),
        help="Target language (English, French, Auto, …) when --voices is not used",
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.environ.get("QWEN_TTS_CHUNK_SIZE", "12")),
        help="Streaming chunk size in codec steps (default: 12)",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=int(os.environ.get("QWEN_TTS_MAX_SEQ_LEN", "512")),
        help="Max decode sequence length for CUDA graph StaticCache (default: 512). "
             "Reduce to save VRAM; 512 ≈ 40 seconds of audio.",
    )
    return p.parse_args()


def main():
    global tts_model, voices, default_voice, SAMPLE_RATE

    args = _parse_args()

    # Build voice registry
    if args.voices:
        with open(args.voices) as f:
            voices = json.load(f)
        default_voice = next(iter(voices))
        logger.info("Loaded %d voice(s) from %s", len(voices), args.voices)
    elif args.ref_audio:
        voices = {
            "default": {
                "ref_audio": args.ref_audio,
                "ref_text": args.ref_text,
                "language": args.language,
                "chunk_size": args.chunk_size,
                "mode": "icl",
            }
        }
        default_voice = "default"
        logger.info("Using single voice from --ref-audio: %s", args.ref_audio)
    else:
        print(
            "ERROR: provide --ref-audio <file> or --voices <config.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    from faster_qwen3_tts import FasterQwen3TTS

    logger.info("Loading model %s on %s (max_seq_len=%d) …", args.model, args.device, args.max_seq_len)
    tts_model = FasterQwen3TTS.from_pretrained(
        args.model,
        device=args.device,
        dtype=torch.bfloat16,
        max_seq_len=args.max_seq_len,
    )
    SAMPLE_RATE = tts_model.sample_rate
    logger.info("Model ready. Sample rate: %d Hz", SAMPLE_RATE)

    # Warmup once so first real request doesn't pay graph capture/prompt cache costs.
    warmup_text = "Hola, prueba rápida de calentamiento del modelo."
    with contextlib.suppress(Exception):
        for _chunk, _sr, _timing in tts_model.generate_voice_clone_streaming(
            text=warmup_text,
            language=voices[default_voice].get("language", "Auto"),
            ref_audio=voices[default_voice]["ref_audio"],
            ref_text=voices[default_voice].get("ref_text", ""),
            chunk_size=voices[default_voice].get("chunk_size", args.chunk_size),
            non_streaming_mode=False,
        ):
            break
    # Release PyTorch allocator cache blocks retained from warmup/graph capture
    # so the first real synthesis request has maximum free VRAM headroom.
    torch.cuda.empty_cache()
    logger.info("Warmup finished")

    logger.info("Server listening on http://%s:%d", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
