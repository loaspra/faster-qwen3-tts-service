#!/usr/bin/env python3
"""
OpenAI-compatible TTS API server for faster-qwen3-tts.

Exposes:
  POST /v1/audio/speech              – OpenAI-compatible one-shot synthesis
  POST /v1/tts/jobs                  – create async TTS job (returns job_id + initial manifest)
  GET  /v1/tts/jobs/{id}             – poll manifest (segment readiness + audio URLs)
  DELETE /v1/tts/jobs/{id}           – cancel job
  GET  /v1/tts/jobs/{id}/segments/{n} – fetch complete WebM/Opus audio for segment n

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

API usage:
    # Create job (text via POST body):
    JOB=$(curl -s -X POST http://localhost:8000/v1/tts/jobs \\
        -H "Content-Type: application/json" \\
        -d '{"text":"Hello world","voice":"default","format":"webm"}')
    JOB_ID=$(echo $JOB | jq -r .job_id)

    # Poll until segment 0 is ready:
    curl -s http://localhost:8000/v1/tts/jobs/$JOB_ID | jq .

    # Fetch segment 0 audio:
    curl -s http://localhost:8000/v1/tts/jobs/$JOB_ID/segments/0 --output seg0.webm
"""
import argparse
import asyncio
import contextlib
import dataclasses
import enum
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
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
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

# ---------------------------------------------------------------------------
# Job registry
# ---------------------------------------------------------------------------

_jobs: dict[str, "TtsJob"] = {}

# TTLs (seconds)
_JOB_TTL_DONE = 120        # 2 min after all segments ready
_JOB_TTL_CANCELLED = 30    # 30s after cancel
_JOB_TTL_UNUSED = 300      # 5 min if never polled after creation


class SegmentStatus(str, enum.Enum):
    queued = "queued"
    synthesizing = "synthesizing"
    ready = "ready"
    failed = "failed"
    cancelled = "cancelled"


class JobStatus(str, enum.Enum):
    running = "running"
    done = "done"
    cancelled = "cancelled"
    failed = "failed"


@dataclasses.dataclass
class SegmentInfo:
    index: int
    text: str
    status: SegmentStatus = SegmentStatus.queued
    audio_bytes: Optional[bytes] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None


@dataclasses.dataclass
class TtsJob:
    job_id: str
    voice: str
    fmt: str                         # "webm" or "wav"
    segments: list[SegmentInfo]
    status: JobStatus = JobStatus.running
    created_at: float = dataclasses.field(default_factory=time.monotonic)
    finished_at: Optional[float] = None
    last_polled_at: Optional[float] = None
    cancel_event: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
    worker_task: Optional[asyncio.Task] = None

    def is_expired(self) -> bool:
        now = time.monotonic()
        if self.status in (JobStatus.done, JobStatus.failed) and self.finished_at:
            return (now - self.finished_at) > _JOB_TTL_DONE
        if self.status == JobStatus.cancelled and self.finished_at:
            return (now - self.finished_at) > _JOB_TTL_CANCELLED
        if self.last_polled_at is None:
            return (now - self.created_at) > _JOB_TTL_UNUSED
        return False


def _gc_jobs():
    expired = [jid for jid, j in _jobs.items() if j.is_expired()]
    for jid in expired:
        del _jobs[jid]
    if expired:
        _log_tts("jobs_gc", removed=len(expired), remaining=len(_jobs))


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _preview_text(text: str, max_chars: int = MAX_LOG_TEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...(+{len(text) - max_chars} chars)"


def _log_tts(event: str, **kwargs):
    logger.info("[tts-service] %s", json.dumps({"event": event, **kwargs}, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Text sanitization + segmentation
# ---------------------------------------------------------------------------

_SEGMENT_MAX_CHARS = 300
_SEGMENT_MIN_CHARS = 120


def sanitize_for_tts(text: str) -> str:
    """Strip markdown and normalise whitespace before synthesis."""
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}([^*\n]+?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}([^_\n]+?)_{1,2}", r"\1", text)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"-{2,}", " - ", text)
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def segment_text(text: str) -> list[str]:
    """Split sanitised text into synthesis-friendly segments."""
    text = sanitize_for_tts(text)
    if not text:
        return []

    paragraphs = [b.strip() for b in re.split(r"\n\n+", text) if b.strip()]

    split_on_sentences: list[str] = []
    for para in paragraphs:
        if len(para) <= _SEGMENT_MAX_CHARS:
            split_on_sentences.append(para)
        else:
            sentences = re.split(r"(?<=[.!?…])\s+", para)
            current = ""
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if current and len(current) + 1 + len(sent) <= _SEGMENT_MAX_CHARS:
                    current = f"{current} {sent}"
                else:
                    if current:
                        split_on_sentences.append(current)
                    current = sent
            if current:
                split_on_sentences.append(current)

    merged: list[str] = []
    for seg in split_on_sentences:
        if merged and (len(merged[-1]) < _SEGMENT_MIN_CHARS or len(seg) < _SEGMENT_MIN_CHARS):
            candidate = f"{merged[-1]} {seg}"
            if len(candidate) <= _SEGMENT_MAX_CHARS:
                merged[-1] = candidate
                continue
        merged.append(seg)

    return merged if merged else [text]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: str = "wav"   # wav | pcm | mp3 | webm
    speed: float = 1.0


class JobCreateRequest(BaseModel):
    text: str
    voice: str = "default"
    format: str = Field(default="webm", description="webm | wav")
    speed: float = 1.0


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _to_pcm16(pcm: np.ndarray) -> bytes:
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def _to_pcm_f32le(pcm: np.ndarray) -> bytes:
    return np.asarray(pcm, dtype=np.float32).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
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
    raw = _to_pcm16(pcm)
    return _wav_header(sample_rate, len(raw)) + raw


def _to_mp3_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
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
# WebM/Opus encoder via ffmpeg (produces a COMPLETE self-contained file)
# ---------------------------------------------------------------------------

async def _pcm_f32le_to_webm_bytes(
    pcm_chunks: list[bytes],
    sample_rate: int = 24000,
    bitrate: str = "24k",
) -> bytes:
    """Encode a collected list of raw f32le PCM chunks into a complete WebM/Opus
    file.  stdin is closed after all chunks are written so ffmpeg can finalize
    the WebM container with proper duration metadata.

    Returns the complete file as bytes.
    """
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
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
    # Write all PCM then close stdin so ffmpeg flushes the complete container
    for chunk in pcm_chunks:
        assert proc.stdin is not None
        proc.stdin.write(chunk)
    proc.stdin.close()
    await proc.stdin.wait_closed()

    stdout, _ = await proc.communicate()
    await proc.wait()
    return stdout


async def _pcm_f32le_to_wav_bytes(
    pcm_chunks: list[bytes],
    sample_rate: int = 24000,
) -> bytes:
    """Collect f32le PCM chunks and produce a complete WAV file."""
    all_pcm = b"".join(pcm_chunks)
    arr = np.frombuffer(all_pcm, dtype=np.float32)
    return _to_wav_bytes(arr, sample_rate)


# ---------------------------------------------------------------------------
# GPU inference helper
# ---------------------------------------------------------------------------


async def _collect_segment_pcm_f32(
    voice_cfg: dict,
    text: str,
    job_id: str,
    seg_idx: int,
) -> list[bytes]:
    """Run generate_voice_clone_streaming for one text segment in a background
    thread, collecting raw float32 PCM chunks.  Returns list[bytes] (f32le).
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
                    q.put(_to_pcm_f32le(chunk))
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(_DONE)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()
    chunks: list[bytes] = []
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _DONE:
            break
        if isinstance(item, Exception):
            raise item
        chunks.append(item)
    return chunks


# ---------------------------------------------------------------------------
# Async synthesis worker
# ---------------------------------------------------------------------------


async def _synthesize_job(job: TtsJob):
    """Background asyncio Task that synthesizes each segment sequentially.

    For each segment:
      1. Acquire model lock (via background thread) and collect PCM chunks.
      2. Encode the PCM into a complete self-contained WebM/Opus (or WAV) file.
      3. Store result in seg.audio_bytes and mark seg.status = ready.

    Checks cancel_event between segments so cancellation is responsive.
    """
    voice_cfg = resolve_voice(job.voice)
    all_ok = True

    for seg in job.segments:
        if job.cancel_event.is_set():
            seg.status = SegmentStatus.cancelled
            continue

        seg.status = SegmentStatus.synthesizing
        started = time.perf_counter()
        _log_tts(
            "job_segment_start",
            job_id=job.job_id,
            segment_index=seg.index,
            segment_count=len(job.segments),
            segment_length=len(seg.text),
            segment_text=_preview_text(seg.text, 200),
        )

        try:
            pcm_chunks = await _collect_segment_pcm_f32(
                voice_cfg, seg.text, job.job_id, seg.index
            )

            if job.fmt == "webm":
                audio_bytes = await _pcm_f32le_to_webm_bytes(pcm_chunks, SAMPLE_RATE)
            else:
                audio_bytes = await _pcm_f32le_to_wav_bytes(pcm_chunks, SAMPLE_RATE)

            # Estimate duration from PCM sample count
            total_samples = sum(len(c) for c in pcm_chunks) // 4  # f32 = 4 bytes
            seg.duration_ms = round(total_samples / SAMPLE_RATE * 1000, 1)
            seg.audio_bytes = audio_bytes
            seg.status = SegmentStatus.ready

            elapsed = round((time.perf_counter() - started) * 1000, 2)
            _log_tts(
                "job_segment_ready",
                job_id=job.job_id,
                segment_index=seg.index,
                audio_bytes=len(audio_bytes),
                duration_ms=seg.duration_ms,
                elapsed_ms=elapsed,
            )

        except Exception as exc:
            seg.status = SegmentStatus.failed
            seg.error = str(exc)
            all_ok = False
            _log_tts(
                "job_segment_failed",
                job_id=job.job_id,
                segment_index=seg.index,
                error=str(exc),
            )

    if job.cancel_event.is_set():
        job.status = JobStatus.cancelled
    elif all_ok:
        job.status = JobStatus.done
    else:
        # Partial failure: mark as done so client can still play ready segments
        job.status = JobStatus.done

    job.finished_at = time.monotonic()
    _log_tts(
        "job_complete",
        job_id=job.job_id,
        status=job.status.value,
        segments_ready=sum(1 for s in job.segments if s.status == SegmentStatus.ready),
        segments_total=len(job.segments),
    )


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------


def resolve_voice(voice_name: str) -> dict:
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
        detail=f"Voice {voice_name!r} not configured. Available: {list(voices.keys())}",
    )


# ---------------------------------------------------------------------------
# Legacy streaming helpers (kept for /v1/audio/speech)
# ---------------------------------------------------------------------------


async def _stream_chunks(
    voice_cfg: dict,
    text: str,
    request_id: str,
) -> AsyncGenerator[tuple[bytes, dict], None]:
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


async def _pcm_to_webm_opus_stream(
    pcm_source: AsyncGenerator[bytes, None],
    sample_rate: int = 24000,
    bitrate: str = "24k",
) -> AsyncGenerator[bytes, None]:
    """Live streaming encoder used only by /v1/audio/speech format=webm."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        "-c:a", "libopus", "-application", "voip", "-b:a", bitrate,
        "-vbr", "on", "-compression_level", "10", "-frame_duration", "20",
        "-f", "webm", "pipe:1",
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


async def _multi_segment_pcm_f32(
    voice_cfg: dict,
    segments: list[str],
    request_id: str,
) -> AsyncGenerator[bytes, None]:
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
# Manifest helper
# ---------------------------------------------------------------------------

_CONTENT_TYPES = {
    "webm": 'audio/webm; codecs="opus"',
    "wav": "audio/wav",
}


def _job_to_manifest(job: TtsJob, base_url: str = "") -> dict:
    segs = []
    for s in job.segments:
        entry: dict = {
            "index": s.index,
            "status": s.status.value,
            "text_length": len(s.text),
        }
        if s.status == SegmentStatus.ready:
            entry["audio_url"] = f"{base_url}/v1/tts/jobs/{job.job_id}/segments/{s.index}"
            entry["duration_ms"] = s.duration_ms
        if s.status == SegmentStatus.failed:
            entry["error"] = s.error
        segs.append(entry)

    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "format": job.fmt,
        "mime": _CONTENT_TYPES.get(job.fmt, "audio/webm"),
        "segments": segs,
        "segments_ready": sum(1 for s in job.segments if s.status == SegmentStatus.ready),
        "segments_total": len(job.segments),
    }


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


# ---- /v1/tts/jobs ----


@app.post("/v1/tts/jobs")
async def create_job(req: JobCreateRequest, request: Request):
    """Create an async TTS job. Returns immediately with job_id and initial
    manifest. Synthesis runs in the background; poll GET /v1/tts/jobs/{id}
    to check segment readiness."""
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    fmt = req.format.lower()
    if fmt not in ("webm", "wav"):
        raise HTTPException(status_code=400, detail=f"format {fmt!r} not supported. Use: webm, wav")

    # Validate voice eagerly
    resolve_voice(req.voice)

    _gc_jobs()

    segments_text = segment_text(text)
    job_id = str(uuid.uuid4())
    job = TtsJob(
        job_id=job_id,
        voice=req.voice,
        fmt=fmt,
        segments=[
            SegmentInfo(index=i, text=s)
            for i, s in enumerate(segments_text)
        ],
    )
    _jobs[job_id] = job

    # Launch background synthesis worker
    job.worker_task = asyncio.create_task(_synthesize_job(job))

    _log_tts(
        "job_created",
        job_id=job_id,
        voice=req.voice,
        format=fmt,
        text_length=len(text),
        segments=len(segments_text),
        segment_lengths=[len(s) for s in segments_text],
    )

    manifest = _job_to_manifest(job)
    manifest["manifest_url"] = f"/v1/tts/jobs/{job_id}"
    manifest["cancel_url"] = f"/v1/tts/jobs/{job_id}"
    return manifest


@app.get("/v1/tts/jobs/{job_id}")
async def get_job_manifest(job_id: str):
    """Poll the job manifest to check segment readiness."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    job.last_polled_at = time.monotonic()
    return _job_to_manifest(job)


@app.delete("/v1/tts/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job. Segments not yet synthesized are marked cancelled."""
    job = _jobs.get(job_id)
    if job is None:
        return {"status": "not_found"}
    job.cancel_event.set()
    job.status = JobStatus.cancelled
    job.finished_at = time.monotonic()
    _log_tts("job_cancelled", job_id=job_id)
    return {"status": "cancelled", "job_id": job_id}


@app.get("/v1/tts/jobs/{job_id}/segments/{segment_index}")
async def get_segment(job_id: str, segment_index: int):
    """Fetch the complete audio file for a ready segment."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or expired")

    if segment_index < 0 or segment_index >= len(job.segments):
        raise HTTPException(status_code=404, detail=f"Segment {segment_index} does not exist")

    seg = job.segments[segment_index]

    if seg.status == SegmentStatus.queued or seg.status == SegmentStatus.synthesizing:
        raise HTTPException(status_code=409, detail=f"Segment {segment_index} not ready yet (status: {seg.status.value})")
    if seg.status == SegmentStatus.failed:
        raise HTTPException(status_code=500, detail=f"Segment {segment_index} synthesis failed: {seg.error}")
    if seg.status == SegmentStatus.cancelled:
        raise HTTPException(status_code=410, detail=f"Segment {segment_index} was cancelled")
    if seg.audio_bytes is None:
        raise HTTPException(status_code=500, detail=f"Segment {segment_index} has no audio data")

    content_type = _CONTENT_TYPES.get(job.fmt, "audio/webm")
    _log_tts(
        "segment_served",
        job_id=job_id,
        segment_index=segment_index,
        bytes=len(seg.audio_bytes),
    )
    return Response(
        content=seg.audio_bytes,
        media_type=content_type,
        headers={
            "Content-Length": str(len(seg.audio_bytes)),
            "Cache-Control": "no-store",
            "X-Job-Id": job_id,
            "X-Segment-Index": str(segment_index),
            "X-Duration-Ms": str(seg.duration_ms or 0),
        },
    )


# ---- /v1/audio/speech (backward-compatible, used by TUI simple path) ----


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

    _log_tts(
        "request_received",
        request_id=request_id,
        voice=req.voice,
        response_format=fmt,
        input_length=len(req.input),
        input_text=_preview_text(req.input),
    )

    all_content_types = {
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "mp3": "audio/mpeg",
        "webm": 'audio/webm; codecs="opus"',
    }
    if fmt not in all_content_types:
        raise HTTPException(
            status_code=400,
            detail=f"response_format {fmt!r} not supported. Use: wav, pcm, mp3, webm",
        )

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
        _log_tts("request_complete_mp3", request_id=request_id,
                 elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2))
        return Response(content=_to_mp3_bytes(audio, sr), media_type=all_content_types["mp3"])

    segments = segment_text(req.input)

    if fmt == "webm":
        pcm_gen = _multi_segment_pcm_f32(voice_cfg, segments, request_id)

        async def webm_stream():
            total_bytes = 0
            first_at = None
            try:
                async for data in _pcm_to_webm_opus_stream(pcm_gen, SAMPLE_RATE):
                    total_bytes += len(data)
                    if first_at is None:
                        first_at = time.perf_counter()
                        _log_tts("stream_ttfa", request_id=request_id,
                                 ttfa_ms=round((first_at - started_at) * 1000, 2))
                    yield data
            finally:
                _log_tts("request_complete_webm", request_id=request_id,
                         total_bytes=total_bytes,
                         elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2))

        resp = StreamingResponse(webm_stream(), media_type=all_content_types["webm"])
        resp.headers["X-Request-Id"] = request_id
        return resp

    # WAV / PCM streaming
    async def audio_stream():
        total_bytes = 0
        first_at = None
        if fmt == "wav":
            header = _wav_header(SAMPLE_RATE)
            total_bytes += len(header)
            yield header
        try:
            for seg in segments:
                async for raw_chunk, _ in _stream_chunks(voice_cfg, seg, request_id):
                    total_bytes += len(raw_chunk)
                    if first_at is None:
                        first_at = time.perf_counter()
                        _log_tts("stream_ttfa", request_id=request_id,
                                 ttfa_ms=round((first_at - started_at) * 1000, 2))
                    yield raw_chunk
        finally:
            _log_tts("request_complete_stream", request_id=request_id,
                     total_bytes=total_bytes,
                     elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2))

    resp = StreamingResponse(audio_stream(), media_type=all_content_types[fmt])
    resp.headers["X-Request-Id"] = request_id
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
    p.add_argument("--model", default=os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
    p.add_argument("--voices", default=os.environ.get("QWEN_TTS_VOICES"), metavar="FILE")
    p.add_argument("--ref-audio", default=os.environ.get("QWEN_TTS_REF_AUDIO"), metavar="FILE")
    p.add_argument("--ref-text", default=os.environ.get("QWEN_TTS_REF_TEXT", ""))
    p.add_argument("--language", default=os.environ.get("QWEN_TTS_LANGUAGE", "Auto"))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk-size", type=int, default=int(os.environ.get("QWEN_TTS_CHUNK_SIZE", "12")))
    p.add_argument(
        "--max-seq-len", type=int,
        default=int(os.environ.get("QWEN_TTS_MAX_SEQ_LEN", "512")),
    )
    return p.parse_args()


def main():
    global tts_model, voices, default_voice, SAMPLE_RATE

    args = _parse_args()

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
        print("ERROR: provide --ref-audio <file> or --voices <config.json>", file=sys.stderr)
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
    torch.cuda.empty_cache()
    logger.info("Warmup finished")
    logger.info("Server listening on http://%s:%d", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
