#!/usr/bin/env python3
"""Évaluation du doublage (sujet D2) : WER, erreur de diarisation, décalage temporel,
écoute à l'aveugle.

  python evaluate.py wer --ref reference.txt --hyp segments.json
  python evaluate.py der --ref reference.rttm --hyp segments.json
  python evaluate.py timing --segments segment_fr.json --audio fr.wav --placed fr.placed.json
  python evaluate.py blind-prepare --system-1 clips_ours --system-2 clips_baseline --out blind
  python evaluate.py blind-score --key blind.key.json --responses r1.csv r2.csv
"""
import argparse
import json
import sys
from pathlib import Path

from evaluation import blind_test, metrics


def read_text(path):
    """Texte brut, ou texte des segments si c'est un JSON / RTTM."""
    if Path(path).suffix.lower() in (".json", ".rttm"):
        return metrics.segments_text(metrics.load_segments(path))
    return Path(path).read_text(encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("wer", help="WER entre transcription de référence et hypothèse")
    p.add_argument("--ref", required=True)
    p.add_argument("--hyp", required=True)

    p = sub.add_parser("der", help="erreur de diarisation (référence .rttm ou .json)")
    p.add_argument("--ref", required=True)
    p.add_argument("--hyp", required=True)
    p.add_argument("--step", type=float, default=0.01, help="pas de la grille en s")

    p = sub.add_parser("timing", help="décalage entre segments prévus et parole doublée")
    p.add_argument("--segments", required=True, help="JSON de segments (starts/ends cibles)")
    p.add_argument("--audio", required=True, help="piste doublée (wav)")
    p.add_argument("--placed", help="<langue>.placed.json écrit par dubbing.py : ajoute débordements exacts et chevauchements")

    p = sub.add_parser("blind-prepare", help="prépare les clips A/B anonymisés")
    p.add_argument("--system-1", required=True)
    p.add_argument("--system-2", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=None)

    p = sub.add_parser("blind-score", help="dépouille les réponses des auditeurs")
    p.add_argument("--key", required=True)
    p.add_argument("--responses", nargs="+", required=True)

    args = ap.parse_args(argv)

    if args.cmd == "wer":
        result = {"wer": metrics.wer(read_text(args.ref), read_text(args.hyp))}
    elif args.cmd == "der":
        result = metrics.diarization_error_rate(
            metrics.load_segments(args.ref), metrics.load_segments(args.hyp), step=args.step)
    elif args.cmd == "timing":
        import soundfile as sf
        audio, sr = sf.read(args.audio, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        segments = metrics.load_segments(args.segments)
        _, result = metrics.timing_offsets(segments, audio, sr)
        if args.placed:
            placed = json.loads(Path(args.placed).read_text(encoding="utf-8"))
            result["placed"] = metrics.placement_overruns(segments, placed)
    elif args.cmd == "blind-prepare":
        key = blind_test.prepare(args.system_1, args.system_2, args.out, seed=args.seed)
        result = {"clips": len(key), "key": f"{args.out}.key.json"}
    else:
        result = blind_test.score(args.key, args.responses)

    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
