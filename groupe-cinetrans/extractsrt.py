"""Étape 1 du pipeline : vidéo -> SRT annoté (locuteur + émotion + langue).

mp4 --ffmpeg--> wav 16 kHz mono
    --WhisperX-->     transcription (faster-whisper) + alignement mot à mot
                      + diarisation (pyannote, intégrée à WhisperX) -> locuteur par segment
    --wav2vec2 SER--> émotion de chaque segment

Modes de langue :
  --source-lang auto   détecte la langue (sur le début de l'audio)   [défaut]
  --source-lang fr     langue imposée
  --multilingual       plusieurs langues parlées dans la vidéo : la langue est
                       détectée puis la transcription refaite pour CHAQUE segment

Tous les modèles tournent en local.
Sorties dans --out-dir : <nom>.srt, <nom>.json, <nom>.wav
"""
import argparse
import json
import os
from pathlib import Path

import ffmpeg  # paquet pip : ffmpeg-python (pas "ffmpeg")
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from transformers import pipeline as hf_pipeline

from diarization_tuning import MIN_CLUSTER_SIZE, SPEAKER_THRESHOLDS, tune_diarizer  # noqa: F401

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")  # tiny, base, small, medium, large-v3-turbo, large-v3
# faster-whisper 1.0.0 (imposé par whisperx 3.2.0) ne connaît pas "large-v3-turbo" : on passe par le dépôt CT2.
WHISPER_ALIASES = {"large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2"}
# Niveau d'émotion -> modèle (de plus en plus lourd). EMOTION_MODEL force un modèle précis.
EMOTION_MODELS = {
    1: "superb/wav2vec2-base-superb-er",  # ~95M params, 4 émotions
    2: "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",  # ~316M, 8 émotions
    3: "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",  # ~637M, 7 émotions
}
EMOTION_LABELS = {  # harmonise les étiquettes entre modèles
    "neu": "neutral", "hap": "happy", "ang": "angry", "fear": "fearful",
    "surprise": "surprised",
}
HF_TOKEN = os.getenv("HF_TOKEN")
SAMPLE_RATE = 16000
MIN_EMOTION_SECONDS = 0.5  # en dessous, le modèle d'émotion n'est pas fiable
MIN_LANG_SECONDS = 1.0  # en dessous, la détection de langue n'est pas fiable


def extract_audio(video_path, wav_path):
    try:
        (
            ffmpeg.input(str(video_path))
            .output(str(wav_path), ac=1, ar=SAMPLE_RATE)
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        raise SystemExit(f"ffmpeg a échoué :\n{e.stderr.decode(errors='replace')}")


def format_timestamp(seconds):
    total_ms = round(seconds * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


class Aligner:
    """Charge le modèle d'alignement une seule fois par langue."""

    def __init__(self, device):
        self.device = device
        self._cache = {}

    def align(self, segments, audio, language):
        if language not in self._cache:
            try:
                self._cache[language] = whisperx.load_align_model(language, self.device)
            except Exception as e:  # langue sans modèle d'alignement
                print(f"  (pas d'alignement pour '{language}': {e})")
                self._cache[language] = None
        loaded = self._cache[language]
        if loaded is None:
            return segments
        model, metadata = loaded
        return whisperx.align(segments, model, metadata, audio, self.device)["segments"]


def transcribe_single(model, audio, aligner, source_lang):
    result = model.transcribe(audio, batch_size=8, language=source_lang)
    language = result["language"]
    segments = aligner.align(result["segments"], audio, language)
    for seg in segments:
        seg["language"] = language
    return segments


def transcribe_multilingual(model, audio, aligner):
    """1re passe : découpe en segments ; 2e passe : langue + transcription par segment."""
    first_pass = model.transcribe(audio, batch_size=8)["segments"]
    last_language = model.detect_language(audio[: 30 * SAMPLE_RATE])
    segments = []
    for seg in first_pass:
        chunk = audio[int(seg["start"] * SAMPLE_RATE) : int(seg["end"] * SAMPLE_RATE)]
        if len(chunk) < MIN_LANG_SECONDS * SAMPLE_RATE:
            # trop court pour détecter la langue : on garde le segment et la langue précédente
            seg["language"] = last_language
            segments.append(seg)
            continue
        last_language = model.detect_language(chunk)
        sub = model.transcribe(chunk, batch_size=8, language=last_language)["segments"]
        if not sub:  # la 2e passe n'a rien trouvé : on garde le texte de la 1re passe
            seg["language"] = last_language
            segments.append(seg)
            continue
        sub = aligner.align(sub, chunk, last_language)
        for s in sub:  # remet les temps (et ceux des mots) dans le repère global
            s["start"] += seg["start"]
            s["end"] += seg["start"]
            for w in s.get("words", []):
                if "start" in w:
                    w["start"] += seg["start"]
                    w["end"] += seg["start"]
            s["language"] = last_language
        segments.extend(sub)
    return segments


def split_by_speaker(segments, min_words=2):
    """Coupe un segment quand le locuteur change au milieu (d'après les mots)."""
    out = []
    for seg in segments:
        words = seg.get("words") or []
        if len({w["speaker"] for w in words if w.get("speaker")}) < 2:
            out.append(seg)
            continue
        runs = []  # [locuteur, [mots]]
        for w in words:
            spk = w.get("speaker") or (runs[-1][0] if runs else seg.get("speaker"))
            if runs and runs[-1][0] == spk:
                runs[-1][1].append(w)
            else:
                runs.append([spk, [w]])
        merged = []  # les séries trop courtes (bruit de diarisation) sont absorbées
        for spk, ws in runs:
            if merged and (merged[-1][0] == spk or len(ws) < min_words):
                merged[-1][1].extend(ws)
            else:
                merged.append([spk, ws])
        if len(merged) < 2:
            out.append(seg)
            continue
        sep = "" if seg.get("language") in ("zh", "ja") else " "
        for spk, ws in merged:
            starts = [w["start"] for w in ws if "start" in w]
            ends = [w["end"] for w in ws if "end" in w]
            out.append({
                **{k: v for k, v in seg.items() if k != "words"},
                "text": sep.join(w["word"] for w in ws),
                "start": starts[0] if starts else seg["start"],
                "end": ends[-1] if ends else seg["end"],
                "speaker": spk,
                "words": ws,
            })
    return out


def transcribe_and_diarize(wav_path, device, source_lang, multilingual, whisper_model,
                           num_speakers=None, min_speakers=None, max_speakers=None,
                           sensitivity=2):
    if not HF_TOKEN:
        raise SystemExit(
            "HF_TOKEN manquant : voir HUGGINGFACE_TOKEN.md (copiez .env.example en .env "
            "et renseignez votre token)."
        )
    audio = whisperx.load_audio(str(wav_path))  # float32, 16 kHz
    compute_type = "float16" if device == "cuda" else "int8"
    model = whisperx.load_model(WHISPER_ALIASES.get(whisper_model, whisper_model), device, compute_type=compute_type)
    aligner = Aligner(device)

    if multilingual:
        segments = transcribe_multilingual(model, audio, aligner)
    else:
        segments = transcribe_single(model, audio, aligner, source_lang)

    diarizer = DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
    tune_diarizer(diarizer.model, sensitivity)
    diarization = diarizer(
        audio, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers
    )
    print(f"  {diarization['speaker'].nunique()} locuteur(s) détecté(s) par pyannote")
    result = whisperx.assign_word_speakers(diarization, {"segments": segments}, fill_nearest=True)
    return audio, split_by_speaker(result["segments"])


def detect_emotions(audio, segments, device_id, level):
    model_name = os.getenv("EMOTION_MODEL") or EMOTION_MODELS[level]
    print(f"  modèle d'émotion : {model_name}")
    classifier = hf_pipeline("audio-classification", model=model_name, device=device_id)
    emotions = []
    for seg in segments:
        chunk = audio[int(seg["start"] * SAMPLE_RATE) : int(seg["end"] * SAMPLE_RATE)]
        if len(chunk) < MIN_EMOTION_SECONDS * SAMPLE_RATE:
            emotions.append("neutral")
            continue
        label = classifier({"raw": chunk, "sampling_rate": SAMPLE_RATE}, top_k=1)[0]["label"]
        emotions.append(EMOTION_LABELS.get(label.lower(), label.lower()))
    return emotions


def to_srt(segments, multilingual):
    blocks = []
    for i, seg in enumerate(segments, start=1):
        tag = f"{seg['speaker']} | {seg['emotion']}"
        if multilingual:
            tag += f" | {seg['language']}"
        blocks.append(
            f"{i}\n"
            f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n"
            f"[{tag}] {seg['text']}\n"
        )
    return "\n".join(blocks)


def main():
    parser = argparse.ArgumentParser(description="Vidéo -> SRT (locuteur + émotion + langue)")
    parser.add_argument("--video", default="videos/input.mp4")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument(
        "--source-lang", default="auto", help="code ISO (fr, en, es…) ou 'auto'"
    )
    parser.add_argument(
        "--multilingual", action="store_true", help="plusieurs langues dans la vidéo"
    )
    parser.add_argument(
        "--target-langs",
        default="",
        help="langues cibles, séparées par des virgules (transmises à l'étape de traduction)",
    )
    parser.add_argument("--whisper-model", default=WHISPER_MODEL)
    parser.add_argument("--emotion-level", type=int, choices=[1, 2, 3], default=2,
                        help="1 léger, 2 standard, 3 précis (plus lourd)")
    parser.add_argument("--num-speakers", type=int, help="nombre exact de locuteurs, si connu")
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--speaker-sensitivity", type=int, choices=[1, 2, 3, 4, 5], default=2,
                        help="1 fusionne les voix proches, 5 les sépare davantage")
    args = parser.parse_args()

    video = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav, srt, js = (out_dir / f"{video.stem}{ext}" for ext in (".wav", ".srt", ".json"))
    source_lang = None if args.source_lang == "auto" else args.source_lang
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[1/3] Extraction audio…")
    extract_audio(video, wav)

    print("[2/3] Transcription + diarisation (WhisperX)…")
    audio, raw = transcribe_and_diarize(
        wav, device, source_lang, args.multilingual, args.whisper_model,
        args.num_speakers, args.min_speakers, args.max_speakers, args.speaker_sensitivity,
    )
    segments = [
        {
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": s["text"].strip(),
            "speaker": s.get("speaker", "UNKNOWN"),
            "language": s.get("language"),
        }
        for s in raw
    ]

    print("[3/3] Détection des émotions…")
    device_id = 0 if device == "cuda" else -1
    for seg, emotion in zip(segments, detect_emotions(audio, segments, device_id, args.emotion_level)):
        seg["emotion"] = emotion

    srt.write_text(to_srt(segments, args.multilingual), encoding="utf-8")
    meta = {
        "video": str(video),
        "source_language": "multi" if args.multilingual else (source_lang or "auto"),
        "detected_languages": sorted({s["language"] for s in segments if s["language"]}),
        "target_languages": [l for l in args.target_langs.split(",") if l],
        "settings": {
            "whisper_model": args.whisper_model,
            "emotion_level": args.emotion_level,
            "num_speakers": args.num_speakers,
            "speaker_sensitivity": args.speaker_sensitivity,
        },
        "segments": segments,
    }
    js.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK -> {srt}, {js}")


if __name__ == "__main__":
    main()
