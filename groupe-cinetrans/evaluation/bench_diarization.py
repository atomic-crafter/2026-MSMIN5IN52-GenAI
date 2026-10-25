"""Banc d'essai de la diarisation : DER de pyannote sous plusieurs réglages.

Dossier de données : pour chaque `x.wav`, un `x.ref.json` (liste {start, end, speaker}).
Un tel jeu s'obtient depuis VoxConverse (annotations CC BY 4.0), p. ex. le dépôt Hugging Face
`diarizers-community/voxconverse`. Les fichiers audio ne sont pas versionnés.

  python -m evaluation.bench_diarization --data voxconverse_sample --sensitivity 2 4
"""
import argparse
import json
import os
from pathlib import Path

from diarization_tuning import tune_diarizer
from evaluation.metrics import diarization_error_rate, load_segments


def run(pipeline, wav, num_speakers=None):
    import soundfile as sf
    import torch
    audio, sr = sf.read(wav, dtype="float32")
    annotation = pipeline({"waveform": torch.from_numpy(audio)[None], "sample_rate": sr},
                           num_speakers=num_speakers)
    return [{"start": t.start, "end": t.end, "speaker": s}
            for t, _, s in annotation.itertracks(yield_label=True)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sensitivity", type=int, nargs="+", default=[2])
    ap.add_argument("--output", help="écrit les résultats en JSON")
    args = ap.parse_args(argv)

    from pyannote.audio import Pipeline
    token = os.environ["HF_TOKEN"]
    pipelines = {"pyannote par défaut": Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)}
    for sens in args.sensitivity:
        tuned = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        tune_diarizer(tuned, sens)
        pipelines[f"réglage projet (sensibilité {sens})"] = tuned

    results = {}
    for wav in sorted(Path(args.data).glob("*.wav")):
        ref = load_segments(wav.with_suffix(".ref.json"))
        n_true = len({s["speaker"] for s in ref})
        for name, pipe in pipelines.items():
            for known in (False, True):
                hyp = run(pipe, wav, n_true if known else None)
                r = diarization_error_rate(ref, hyp)
                label = f"{name}{' + nb. de locuteurs connu' if known else ''}"
                r["speakers_found"] = len({s["speaker"] for s in hyp})
                r["speakers_true"] = n_true
                results.setdefault(label, {})[wav.stem] = r
                print(f"{wav.stem:8s} {label:55s} DER {r['der']:.3f}  locuteurs {r['speakers_found']}/{n_true}", flush=True)

    print("\nDER moyen (pondéré par la durée de parole) :")
    for label, per_file in results.items():
        speech = sum(r["speech"] for r in per_file.values())
        err = sum(r["der"] * r["speech"] for r in per_file.values())
        print(f"  {label:55s} {err / speech:.3f}")
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
