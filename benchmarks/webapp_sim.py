#!/usr/bin/env python3
"""Webapp-simulation test for the TTS service.

Mimics exactly what the Next.js webapp proxy does: POST /v1/audio/speech with
format=webm, streaming the response. Measures the metrics that matter for the
chat UX and validates that the post-text noise-tail bug is gone.

It does NOT require numpy/soundfile — it shells out to ffmpeg/ffprobe (already
present in the TTS image) to decode WebM/Opus to raw PCM and analyse it.

Metrics per request:
  * TTFB      : time to first response byte (proxy upstream_response_ready)
  * TTFA      : time to first audio byte of the stream
  * total_ms  : wall time until stream completes
  * audio_s   : decoded audio duration
  * rtf       : audio_s / (total_ms/1000)   (>1 means faster than real time)
  * trailing_silence_s : silence at the very end (should be small & quiet)
  * tail_noise: heuristic — is there LOUD audio in the final window that is NOT
                trailing silence? (the 30s+ "weird noise" symptom)
  * peak/rms  : amplitude sanity (clipping detection)

Usage:
  python benchmarks/webapp_sim.py \
      --url http://100.74.7.103:30881 \
      --runs 3 --out /tmp/opencode/webapp_sim

  # localhost against a freshly started server:
  python benchmarks/webapp_sim.py --url http://127.0.0.1:8880
"""
import argparse
import array
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
import uuid

# Texts mirror the real production payloads seen in the webapp logs: long
# Spanish prose, markdown, headers, lists, numbers/units — the cases that
# triggered the noise tails.
DEFAULT_TEXTS = [
    # short — most likely to exhibit a long hallucinated tail relative to speech
    "Vamos, una última repetición y terminamos por hoy.",
    # medium prose
    "El Parque Kennedy un viernes por la noche es el punto de respawn de los "
    "NPCs. Míralos bien mientras caminas hacia el Malecón.",
    # long with markdown + header + list (server strips markdown)
    "**Modo Yeti / Arquitecto de Alto Rendimiento:**\n\n"
    "Acabas de describir la diferencia entre el uno por ciento y el noventa y "
    "nueve. Tú cobras siete mil quinientos dólares, tienes el motor cargado, y "
    "estás calentando para romper tus propios límites.\n\n"
    "**1. Respira.** **2. Enfócate.** **3. Ejecuta.**",
]

SAMPLE_RATE = 24000  # service output rate


def percentile(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def stream_tts(base_url, text, fmt="webm", voice="default", speed=1.0):
    """POST /v1/audio/speech and stream the body, mirroring the webapp proxy.

    Returns (audio_bytes, metrics_dict).
    """
    req_id = str(uuid.uuid4())
    body = json.dumps({
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "voice": voice,
        "input": text,
        "response_format": fmt,
        "speed": speed,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/audio/speech",
        data=body,
        headers={"Content-Type": "application/json", "X-Request-Id": req_id},
        method="POST",
    )
    t0 = time.perf_counter()
    ttfb = ttfa = None
    chunks = []
    with urllib.request.urlopen(req, timeout=300) as resp:
        ttfb = (time.perf_counter() - t0) * 1000
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            if ttfa is None:
                ttfa = (time.perf_counter() - t0) * 1000
            chunks.append(chunk)
    total_ms = (time.perf_counter() - t0) * 1000
    data = b"".join(chunks)
    return data, {
        "request_id": req_id,
        "text_len": len(text),
        "ttfb_ms": round(ttfb, 1),
        "ttfa_ms": round(ttfa, 1) if ttfa else None,
        "total_ms": round(total_ms, 1),
        "bytes": len(data),
    }


def decode_to_pcm_s16(audio_bytes, fmt):
    """Decode any container (webm/wav/mp3) to mono s16le PCM via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", fmt if fmt in ("wav",) else fmt, "-i", "pipe:0",
         "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
        input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        # retry letting ffmpeg sniff the format
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
            input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode()[:400]}")
    return array.array("h", proc.stdout)


def analyse_pcm(pcm):
    """Return amplitude + tail-noise metrics from mono s16le samples."""
    n = len(pcm)
    if n == 0:
        return {"audio_s": 0.0, "empty": True}

    peak = 0
    sumsq = 0.0
    clip_count = 0
    for s in pcm:
        a = abs(s)
        if a > peak:
            peak = a
        sumsq += float(s) * float(s)
        if a >= 32767:
            clip_count += 1
    rms = math.sqrt(sumsq / n)

    # Frame-level RMS (20ms frames) for silence/tail-noise analysis.
    frame = int(SAMPLE_RATE * 0.02)
    frames_rms = []
    for i in range(0, n, frame):
        seg = pcm[i:i + frame]
        if not seg:
            continue
        ss = 0.0
        for s in seg:
            ss += float(s) * float(s)
        frames_rms.append(math.sqrt(ss / len(seg)))

    # Trailing silence: count quiet frames at the end. Quiet = rms < 2% FS.
    quiet_thresh = 0.02 * 32767
    trailing_quiet = 0
    for r in reversed(frames_rms):
        if r < quiet_thresh:
            trailing_quiet += 1
        else:
            break
    trailing_silence_s = trailing_quiet * 0.02

    # Tail-noise heuristic (the 30s+ "weird loud noise" symptom):
    # find the last "loud" frame (rms > 8% FS, i.e. real speech), then measure
    # how much LOUD audio exists AFTER what should have been the natural end.
    speech_thresh = 0.08 * 32767
    last_loud = -1
    for idx, r in enumerate(frames_rms):
        if r > speech_thresh:
            last_loud = idx
    audio_s = n / SAMPLE_RATE

    # If there is a big chunk of non-quiet audio with NO speech-level energy at
    # the tail, that's the hallucinated-noise signature. We approximate by:
    # tail region = everything after last_loud frame. Within it, count frames
    # that are neither quiet (silence) nor loud (speech) — "buzzy" mid-energy
    # frames are the hallucination.
    mid_lo, mid_hi = quiet_thresh, speech_thresh
    tail_buzz_frames = 0
    if last_loud >= 0:
        for r in frames_rms[last_loud + 1:]:
            if mid_lo <= r < mid_hi:
                tail_buzz_frames += 1
    tail_buzz_s = tail_buzz_frames * 0.02

    return {
        "audio_s": round(audio_s, 2),
        "peak_pct": round(peak / 32767 * 100, 1),
        "rms_pct": round(rms / 32767 * 100, 1),
        "clip_samples": clip_count,
        "trailing_silence_s": round(trailing_silence_s, 2),
        "tail_buzz_s": round(tail_buzz_s, 2),
        "n_frames": len(frames_rms),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("TTS_URL", "http://100.74.7.103:30881"))
    ap.add_argument("--runs", type=int, default=3, help="repeats per text (consistency)")
    ap.add_argument("--format", default="webm", choices=["webm", "wav", "mp3"])
    ap.add_argument("--out", default="/tmp/opencode/webapp_sim", help="output dir for audio + report")
    ap.add_argument("--text", action="append", help="override text (repeatable)")
    args = ap.parse_args()

    texts = args.text if args.text else DEFAULT_TEXTS
    os.makedirs(args.out, exist_ok=True)

    print(f"TTS webapp-sim against {args.url} | format={args.format} | runs={args.runs}\n")
    print(f"{'text#':<6}{'run':<5}{'ttfb':>8}{'ttfa':>8}{'total':>9}"
          f"{'audio':>8}{'rtf':>7}{'peak%':>7}{'clip':>7}{'tailSil':>9}{'tailBuzz':>10}")
    print("-" * 90)

    rows = []
    for ti, text in enumerate(texts):
        for run in range(args.runs):
            try:
                data, m = stream_tts(args.url, text, fmt=args.format)
            except Exception as exc:
                print(f"{ti:<6}{run:<5}  ERROR: {exc}")
                rows.append({"text_index": ti, "run": run, "error": str(exc)})
                continue
            try:
                pcm = decode_to_pcm_s16(data, args.format)
                a = analyse_pcm(pcm)
            except Exception as exc:
                print(f"{ti:<6}{run:<5}  DECODE ERROR: {exc}")
                rows.append({"text_index": ti, "run": run, **m, "decode_error": str(exc)})
                continue

            rtf = (a["audio_s"] / (m["total_ms"] / 1000)) if m["total_ms"] else 0
            print(f"{ti:<6}{run:<5}{m['ttfb_ms']:>8.0f}"
                  f"{(m['ttfa_ms'] or 0):>8.0f}{m['total_ms']:>9.0f}"
                  f"{a['audio_s']:>8.2f}{rtf:>7.2f}{a['peak_pct']:>7.1f}"
                  f"{a['clip_samples']:>7}{a['trailing_silence_s']:>9.2f}"
                  f"{a['tail_buzz_s']:>10.2f}")

            # Save the first run of each text for manual listening.
            if run == 0:
                fn = os.path.join(args.out, f"text{ti}_run{run}.{args.format}")
                with open(fn, "wb") as f:
                    f.write(data)
            rows.append({"text_index": ti, "run": run, **m, **a, "rtf": round(rtf, 3)})

    # Aggregate consistency / summary
    ok = [r for r in rows if "error" not in r and "decode_error" not in r and r.get("audio_s")]
    print("\n=== Summary ===")
    if ok:
        ttfas = [r["ttfa_ms"] for r in ok if r.get("ttfa_ms")]
        totals = [r["total_ms"] for r in ok]
        tailbuzz = [r["tail_buzz_s"] for r in ok]
        clips = [r["clip_samples"] for r in ok]
        print(f"  requests ok        : {len(ok)}/{len(rows)}")
        print(f"  TTFA p50/p90 (ms)  : {percentile(ttfas,0.5):.0f} / {percentile(ttfas,0.9):.0f}")
        print(f"  total p50/p90 (ms) : {percentile(totals,0.5):.0f} / {percentile(totals,0.9):.0f}")
        print(f"  max tail_buzz (s)  : {max(tailbuzz):.2f}   <-- noise-tail bug if >> 0")
        print(f"  max clip_samples   : {max(clips)}   <-- hard-clip bangs if >> 0")
        worst = max(ok, key=lambda r: r["tail_buzz_s"])
        if worst["tail_buzz_s"] > 1.0:
            print(f"  !! WARNING: text#{worst['text_index']} run{worst['run']} has "
                  f"{worst['tail_buzz_s']:.1f}s of post-speech noise tail")
    else:
        print("  no successful requests")

    report = os.path.join(args.out, "report.json")
    with open(report, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nAudio + report written to {args.out}")
    print(f"  report: {report}")


if __name__ == "__main__":
    main()
