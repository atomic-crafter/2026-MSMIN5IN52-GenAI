#!/usr/bin/env python3
"""
Dubbing pipeline : JSON annoté -> voix Kokoro -> vidéo doublée.
Lit un JSON produit par translate_json.py (format avec "video", "segments", "target_languages").
"""
import argparse
import json
import math
import subprocess
import sys
import os
import numpy as np
import ffmpeg
import soundfile as sf
from pathlib import Path

SR = 24000  # Kokoro génère toujours du 24 kHz

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

LANG_MAP = {
    "en": "en-us", "es": "es-es", "fr": "fr-fr", "de": "de-de",
    "it": "it-it", "pt": "pt-br", "vi": "vi-vn", "sv": "sv-se",
    "ru": "ru-ru", "hi": "hi-in", "ko": "ko-kr", "ja": "ja-jp",
    "th": "th-th", "cs": "cs-cz", "zh": "zh", "cmn": "zh",
}

VOICES = {
    "en-us": ["af_heart", "am_michael", "af_bella", "am_adam"],
    "fr-fr": ["ff_siwis"],
    "es-es": ["ef_dora", "em_alex"],
    "de-de": ["bf_emma", "bm_george"],
    "it-it": ["if_sara", "im_nicola"],
    "pt-br": ["pf_dora", "pm_alex"],
    "ja-jp": ["jf_alpha", "jm_kumo"],
    "zh": ["zf_001", "zm_009"],  # modèle kokoro v1.1-zh : voix numérotées
    "ko-kr": ["af_heart", "am_michael"],
    "hi-in": ["hf_alpha", "hm_omega"],
    "th-th": ["bf_emma", "bm_george"],
    "cs-cz": ["ef_dora", "em_alex"],
    "sv-se": ["af_bella", "am_adam"],
    "ru-ru": ["bf_isabella", "bm_lewis"],
    "vi-vn": ["af_nicole", "am_fenrir"],
}

EMOTION_SPEED = {
    "angry": 1.08, "anger": 1.08, "happy": 1.05, "happiness": 1.05, "joy": 1.05,
    "surprise": 1.05, "surprised": 1.05, "fear": 1.05, "sad": 0.92,
    "sadness": 0.92, "disgust": 0.97, "neutral": 1.0,
}

MIN_SPEED, MAX_SPEED = 0.8, 1.4
MAX_STRETCH = 1.25
GAP = 0.05
MIN_SLOT = 0.30
MAX_PITCH_SHIFT = 3.0
MAX_GAIN_DB = 6.0
MIN_VOICED_FRAMES = 5


def warn(msg):
    print(f"[WARN] {msg}", file=sys.stderr)


def load_json(json_path):
    """Lit le JSON et retourne {video_path, segments, target_languages}."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Gérer à la fois le format wraps ("video", "segments") et le format plat
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        return {
            "video_path": data["video"],
            "segments": data["segments"],
            "languages": data.get("target_languages", ["fr"]),
        }
    elif isinstance(data, list):
        return {
            "video_path": None,
            "segments": data,
            "languages": ["fr"],
        }
    else:
        sys.exit("Format inattendu : liste de segments ou objet avec 'segments'")


def get_speaker(segments, language):
    """Map chaque speaker à une voix Kokoro pour la langue donnée."""
    kokoro_lang = LANG_MAP.get(language, "en-us")
    voices = VOICES.get(kokoro_lang, VOICES["en-us"])
    speaker_voices = {}
    for segment in segments:
        spk = segment["speaker"]
        if spk not in speaker_voices:
            # Alterner entre les voix disponibles
            idx = len(speaker_voices) % len(voices)
            speaker_voices[spk] = voices[idx]
    return speaker_voices


def get_video_duration(video_path):
    info = ffmpeg.probe(str(video_path))
    return float(info["format"]["duration"])


def to_array(rendered):
    audio = np.asarray(rendered.audio, dtype=np.float32)
    if rendered.sample_rate != SR:
        raise ValueError(f"sample_rate={rendered.sample_rate}, attendu {SR}")
    return audio


def create_items(segments, voice_map, lang):
    """Créer les items pour Kokoro à partir des segments traduits."""
    items = []
    for i, segment in enumerate(segments):
        kokoro_lang = LANG_MAP.get(lang, "en-us")
        voice = voice_map.get(segment["speaker"], VOICES.get(kokoro_lang, ["af_heart"])[0])
        items.append({
            "id": i,
            "text": segment["text"],
            "language": kokoro_lang,
            "voice": voice,
        })
    return items


def get_segment(items, lang, speed=1.0):
    """Synthétise les segments avec Kokoro."""
    from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig, SynthesisSegment
    config = SynthesisConfig(generation=GenerationConfig(lang=lang, speed=speed),
                             long_text_split="sentence")
    requests = []
    for item in items:
        requests.append(SynthesisSegment(
            id=str(item["id"]), text=item["text"],
            language=item["language"], voice=item["voice"]))
    with KokoroSynthesizer(config) as synthesizer:
        rendered_results = [to_array(s) for s in synthesizer.synthesize_segments(requests)]
    return rendered_results


def trim_silence(audio, threshold_db=-40.0, keep=0.02):
    """Retire le silence de début et de fin (Kokoro ajoute ~0,3 s en tête, ce qui décale chaque réplique)."""
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak == 0.0:
        return audio
    idx = np.flatnonzero(np.abs(audio) > peak * 10 ** (threshold_db / 20))
    pad = int(keep * SR)
    return audio[max(idx[0] - pad, 0): idx[-1] + pad + 1]


def get_max_lengths(segments, total_duration):
    """Durée maximale autorisée par segment."""
    max_lengths = []
    for i, seg in enumerate(segments):
        slot = seg["end"] - seg["start"]
        next_start = segments[i + 1]["start"] if i + 1 < len(segments) else total_duration
        max_lengths.append(max(slot, next_start - seg["start"]))
    return max_lengths


def fit_audios(items, audios, limits):
    """Re-synthétise plus vite les segments qui dépassent leur limite, puis tronque."""
    audios = list(audios)
    groups = {}
    for i, audio in enumerate(audios):
        dur = len(audio) / SR
        if dur > limits[i]:
            speed = min(MAX_SPEED, math.ceil(dur / limits[i] * 10) / 10)
            groups.setdefault(speed, []).append(i)

    for speed, indices in groups.items():
        new_audios = get_segment([items[i] for i in indices],
                                 lang=items[0]["language"], speed=speed)
        for i, new_aud in zip(indices, new_audios):
            audios[i] = trim_silence(new_aud)

    fade = int(0.03 * SR)
    for i, audio in enumerate(audios):
        max_samples = int(limits[i] * SR)
        if len(audio) > max_samples:
            audio = audio[:max_samples].copy()
            audio[-fade:] *= np.linspace(1, 0, min(fade, len(audio)))
            audios[i] = audio
    return audios


def build_track(segments, audios, total_duration):
    """Place chaque son à start * SR sur une piste de la longueur de la vidéo."""
    track = np.zeros(int(total_duration * SR), dtype=np.float32)
    for seg, audio in zip(segments, audios):
        start = int(seg["start"] * SR)
        if start >= len(track):
            continue
        end = min(start + len(audio), len(track))
        track[start:end] += audio[: end - start]
    return np.clip(track, -1.0, 1.0)


def mux(video_path, wav_path, out_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path), "-i", str(wav_path),
        "-map", "0:v:0", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path)],
        check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser(description="Dubbing pipeline : JSON -> vidéo doublée")
    ap.add_argument("input_json", help="JSON de segments (produit par translate_json.py)")
    ap.add_argument("--output-dir", "-o", default=None, help="dossier de sortie")
    ap.add_argument("--video", "-v", default=None, help="chemin vidéo (si pas dans le JSON)")
    ap.add_argument("--baseline", action="store_true",
                    help="baseline d'évaluation : une seule voix, aucun ajustement de durée")
    args = ap.parse_args()

    info = load_json(args.input_json)
    base_dir = Path(args.input_json).resolve().parent
    video_path = Path(args.video) if args.video else base_dir / info["video_path"].lstrip("/")

    for langue in info["languages"]:
        print(f"\n=== Dubbing [{langue}] ===")
        speakers = get_speaker(info["segments"], langue)
        if args.baseline:
            first_voice = VOICES.get(LANG_MAP.get(langue, "en-us"), VOICES["en-us"])[0]
            speakers = {spk: first_voice for spk in speakers}
        items = create_items(info["segments"], voice_map=speakers, lang=langue)
        print(f"  {len(items)} segments à synthétiser...")

        audios = [trim_silence(a) for a in get_segment(items=items, lang=LANG_MAP.get(langue, "en-us"))]
        max_lengths = get_max_lengths(info["segments"], get_video_duration(video_path))
        if not args.baseline:
            audios = fit_audios(items=items, audios=audios, limits=max_lengths)
        track = build_track(segments=info["segments"], audios=audios,
                            total_duration=get_video_duration(video_path))

        out_dir = Path(args.output_dir) if args.output_dir else base_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        suffix = "_baseline" if args.baseline else ""
        # Durées réellement placées : permet de mesurer débordements et chevauchements (evaluate.py timing --placed)
        placed = [{"start": seg["start"], "end": seg["start"] + len(a) / SR}
                  for seg, a in zip(info["segments"], audios)]
        (out_dir / f"{langue}{suffix}.placed.json").write_text(json.dumps(placed), encoding="utf-8")

        wav_name = out_dir / f"{langue}{suffix}.wav"
        sf.write(str(wav_name), track, SR)
        print(f"  Audio écrit : {wav_name}")

        new_video = out_dir / f"{langue}{suffix}.mp4"
        mux(video_path, wav_name, new_video)
        print(f"  Vidéo doublée : {new_video}")

    print("\nTerminé !")


if __name__ == "__main__":
    main()
