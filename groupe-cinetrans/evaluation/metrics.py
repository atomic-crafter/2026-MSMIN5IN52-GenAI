"""Métriques du sujet D2 : WER, erreur de diarisation (DER) et décalage temporel.

Les segments sont des dicts {"start", "end", "speaker", "text"} comme dans les JSON
produits par extractsrt.py / translate_json.py.
"""
import itertools
import json
import re
import unicodedata
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# Lecture
# ----------------------------------------------------------------------------

def load_segments(path):
    """Segments depuis un JSON (liste ou objet avec "segments") ou un RTTM."""
    path = Path(path)
    if path.suffix.lower() == ".rttm":
        segments = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start, dur = float(parts[3]), float(parts[4])
                segments.append({"start": start, "end": start + dur, "speaker": parts[7]})
        return segments
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["segments"] if isinstance(data, dict) else data


def segments_text(segments):
    return " ".join(str(s.get("text", "")) for s in segments)


# ----------------------------------------------------------------------------
# WER
# ----------------------------------------------------------------------------

def normalize_text(text):
    """Minuscules, sans accents ni ponctuation : le WER ne doit pas punir la typographie."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^\w\s']|_", " ", text).split()


def wer(reference, hypothesis):
    """Word error rate = (S + D + I) / N, via distance de Levenshtein sur les mots."""
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


# ----------------------------------------------------------------------------
# Diarisation (DER)
# ----------------------------------------------------------------------------

def _labels_on_grid(segments, grid):
    """Pour chaque pas de la grille, l'ensemble des locuteurs actifs."""
    active = [set() for _ in grid]
    for seg in segments:
        lo = np.searchsorted(grid, seg["start"], side="left")
        hi = np.searchsorted(grid, seg["end"], side="left")
        for k in range(lo, hi):
            active[k].add(seg["speaker"])
    return active


def diarization_error_rate(reference, hypothesis, step=0.01):
    """DER sur une grille de `step` secondes, avec la meilleure correspondance
    locuteurs hypothèse -> référence (essai de toutes les permutations).

    Retourne {"der", "missed", "false_alarm", "confusion", "speech"} (durées en s).
    Pas de marge de tolérance (collar) : mesure stricte.
    """
    if not reference:
        raise ValueError("référence vide")
    end = max(s["end"] for s in list(reference) + list(hypothesis))
    grid = np.arange(0.0, end, step)
    ref, hyp = _labels_on_grid(reference, grid), _labels_on_grid(hypothesis, grid)

    ref_spk = sorted({s["speaker"] for s in reference})
    hyp_spk = sorted({s["speaker"] for s in hypothesis})
    co = {(r, h): 0 for r in ref_spk for h in hyp_spk}  # temps de recouvrement
    for r_set, h_set in zip(ref, hyp):
        for r in r_set:
            for h in h_set:
                co[(r, h)] += 1

    # Correspondance maximisant le recouvrement (permutations ; peu de locuteurs).
    best, best_map = -1, {}
    pad = [None] * max(0, len(ref_spk) - len(hyp_spk))
    for perm in set(itertools.permutations(hyp_spk + pad, len(ref_spk))):
        score = sum(co[(r, h)] for r, h in zip(ref_spk, perm) if h is not None)
        if score > best:
            best, best_map = score, {h: r for r, h in zip(ref_spk, perm) if h is not None}

    missed = false_alarm = confusion = speech = 0
    for r_set, h_set in zip(ref, hyp):
        n_ref, n_hyp = len(r_set), len(h_set)
        speech += n_ref
        correct = len({best_map.get(h) for h in h_set} & r_set)
        missed += max(0, n_ref - n_hyp)
        false_alarm += max(0, n_hyp - n_ref)
        confusion += min(n_ref, n_hyp) - correct
    if speech == 0:
        raise ValueError("référence sans parole")
    out = {"missed": missed * step, "false_alarm": false_alarm * step,
           "confusion": confusion * step, "speech": speech * step}
    out["der"] = (missed + false_alarm + confusion) / speech
    return out


# ----------------------------------------------------------------------------
# Décalage temporel du doublage
# ----------------------------------------------------------------------------

def speech_bounds(audio, sr, lo, hi, threshold_db=-35.0, frame=0.01):
    """(début, fin) de la parole dans audio[lo:hi] (secondes absolues), ou None.

    Un cadre est "actif" si son énergie dépasse `threshold_db` sous le pic global.
    """
    peak = float(np.max(np.abs(audio))) or 1.0
    thr = peak * 10 ** (threshold_db / 20)
    n = int(frame * sr)
    a, b = int(max(lo, 0) * sr), int(min(hi * sr, len(audio)))
    frames = [(a + k, np.sqrt(np.mean(audio[a + k:a + k + n] ** 2)))
              for k in range(0, max(0, b - a - n), n)]
    active = [pos for pos, rms in frames if rms > thr]
    if not active:
        return None
    return active[0] / sr, (active[-1] + n) / sr


def timing_offsets(segments, audio, sr, search_margin=0.5):
    """Compare chaque segment prévu à la parole réellement présente dans la piste doublée.

    Pour le segment i, la recherche s'arrête au début du suivant (sinon on mesurerait
    la voix du segment suivant). Retourne la liste par segment et un résumé :
    décalage de début (signé, s), dépassement de fin (s, >0 = déborde) et segments muets.
    """
    rows = []
    for i, seg in enumerate(segments):
        nxt = segments[i + 1]["start"] if i + 1 < len(segments) else len(audio) / sr
        # Ne pas remonter dans le segment précédent : la marge est limitée à la moitié
        # de l'intervalle qui le sépare de celui-ci.
        gap = seg["start"] - segments[i - 1]["end"] if i else search_margin
        margin = min(search_margin, max(gap, 0) / 2)
        bounds = speech_bounds(audio, sr, seg["start"] - margin, max(seg["end"], nxt))
        if bounds is None:
            rows.append({"index": i, "silent": True})
            continue
        rows.append({"index": i, "silent": False,
                     "start_offset": bounds[0] - seg["start"],
                     "end_overrun": bounds[1] - seg["end"]})
    voiced = [r for r in rows if not r["silent"]]
    if not voiced:
        return rows, {"segments": len(rows), "silent": len(rows)}
    start = np.array([r["start_offset"] for r in voiced])
    over = np.array([r["end_overrun"] for r in voiced])
    return rows, {
        "segments": len(rows),
        "silent": len(rows) - len(voiced),
        "mean_abs_start_offset": float(np.mean(np.abs(start))),
        "p95_abs_start_offset": float(np.percentile(np.abs(start), 95)),
        "mean_end_overrun": float(np.mean(over)),
        "pct_overrun_over_0.2s": float(np.mean(over > 0.2) * 100),
    }


def placement_overruns(segments, placed):
    """Débordement exact à partir des durées placées par dubbing.py (fichier *.placed.json).

    end_overrun = fin placée - fin cible ; overlap = temps de parole qui empiète sur le segment suivant.
    Contrairement à timing_offsets, voit aussi les répliques qui se chevauchent.
    """
    if len(segments) != len(placed):
        raise ValueError("segments et durées placées de longueurs différentes")
    over, overlap = [], []
    for i, (seg, pl) in enumerate(zip(segments, placed)):
        over.append(pl["end"] - seg["end"])
        overlap.append(max(0.0, pl["end"] - segments[i + 1]["start"]) if i + 1 < len(segments) else 0.0)
    over, overlap = np.array(over), np.array(overlap)
    return {
        "mean_end_overrun": float(over.mean()),
        "pct_overrun_over_0.2s": float(np.mean(over > 0.2) * 100),
        "pct_overlapping_next": float(np.mean(overlap > 0.0) * 100),
        "total_overlap_s": float(overlap.sum()),
    }
