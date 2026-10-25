import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import blind_test, metrics  # noqa: E402
import evaluate  # noqa: E402

SR = 24000


def seg(start, end, speaker="A", text=""):
    return {"start": start, "end": end, "speaker": speaker, "text": text}


# --- WER ---------------------------------------------------------------

def test_wer_identical_ignores_case_accents_punctuation():
    assert metrics.wer("Élève, ça va ?", "eleve ca va") == 0.0


def test_wer_substitution_deletion_insertion():
    assert metrics.wer("a b c d", "a x c d") == 0.25
    assert metrics.wer("a b c d", "a b c") == 0.25
    assert metrics.wer("a b c d", "a b c d e") == 0.25


def test_wer_empty_cases():
    assert metrics.wer("", "") == 0.0
    assert metrics.wer("", "x") == 1.0
    assert metrics.wer("a b", "") == 1.0


# --- DER ---------------------------------------------------------------

def test_der_perfect_even_with_renamed_speakers():
    ref = [seg(0, 5, "A"), seg(5, 10, "B")]
    hyp = [seg(0, 5, "spk1"), seg(5, 10, "spk0")]
    assert metrics.diarization_error_rate(ref, hyp)["der"] == pytest.approx(0, abs=1e-6)


def test_der_confusion():
    ref = [seg(0, 5, "A"), seg(5, 10, "B")]
    hyp = [seg(0, 5, "x"), seg(5, 7, "y"), seg(7, 10, "x")]
    r = metrics.diarization_error_rate(ref, hyp)
    assert r["confusion"] == pytest.approx(3, abs=0.05)
    assert r["der"] == pytest.approx(0.3, abs=0.01)


def test_der_missed_and_false_alarm():
    ref = [seg(0, 10, "A")]
    hyp = [seg(2, 12, "A")]
    r = metrics.diarization_error_rate(ref, hyp)
    assert r["missed"] == pytest.approx(2, abs=0.05)
    assert r["false_alarm"] == pytest.approx(2, abs=0.05)
    assert r["der"] == pytest.approx(0.4, abs=0.01)


def test_der_rejects_empty_reference():
    with pytest.raises(ValueError):
        metrics.diarization_error_rate([], [seg(0, 1)])


def test_load_rttm(tmp_path):
    f = tmp_path / "r.rttm"
    f.write_text("SPEAKER file 1 1.50 2.00 <NA> <NA> alice <NA> <NA>\n")
    assert metrics.load_segments(f) == [{"start": 1.5, "end": 3.5, "speaker": "alice"}]


# --- Décalage temporel ---------------------------------------------------

def track_with_tones(total, tones):
    audio = np.zeros(int(total * SR), dtype=np.float32)
    for start, end in tones:
        t = np.arange(int((end - start) * SR)) / SR
        audio[int(start * SR):int(start * SR) + len(t)] = 0.5 * np.sin(2 * np.pi * 220 * t)
    return audio


def test_timing_on_time_and_late_and_overrun():
    segments = [seg(1.0, 2.0), seg(4.0, 5.0)]
    audio = track_with_tones(8, [(1.0, 2.0), (4.3, 5.6)])
    rows, summary = metrics.timing_offsets(segments, audio, SR)
    assert rows[0]["start_offset"] == pytest.approx(0, abs=0.03)
    assert rows[1]["start_offset"] == pytest.approx(0.3, abs=0.03)
    assert rows[1]["end_overrun"] == pytest.approx(0.6, abs=0.03)
    assert summary["silent"] == 0
    assert summary["pct_overrun_over_0.2s"] == 50.0


def test_timing_adjacent_segments_not_confused_with_previous_speech():
    # Segments contigus : la parole du premier ne doit pas être comptée comme début du second.
    segments = [seg(1.0, 2.0), seg(2.02, 3.0)]
    audio = track_with_tones(5, [(1.0, 2.0), (2.02, 3.0)])
    rows, _ = metrics.timing_offsets(segments, audio, SR)
    assert abs(rows[1]["start_offset"]) < 0.03


def test_timing_silent_segment_is_reported():
    rows, summary = metrics.timing_offsets([seg(1, 2), seg(4, 5)], track_with_tones(8, [(1, 2)]), SR)
    assert rows[1]["silent"] and summary["silent"] == 1


# --- Écoute à l'aveugle --------------------------------------------------

def test_sign_test_p():
    assert blind_test.sign_test_p(0, 0) == 1.0
    assert blind_test.sign_test_p(5, 5) == 1.0
    assert blind_test.sign_test_p(10, 0) == pytest.approx(2 / 1024)


def test_blind_prepare_and_score(tmp_path):
    for d, content in (("s1", b"one"), ("s2", b"two")):
        (tmp_path / d).mkdir()
        for clip in ("c1", "c2", "c3"):
            (tmp_path / d / f"{clip}.mp4").write_bytes(content + clip.encode())
    key = blind_test.prepare(tmp_path / "s1", tmp_path / "s2", tmp_path / "blind", seed=1)
    assert (tmp_path / "blind.key.json").exists()
    assert not (tmp_path / "blind" / "blind.key.json").exists()  # clé hors du dossier distribué
    for clip, labels in key.items():
        assert set(labels) == {"A", "B"} and set(labels.values()) == {"system_1", "system_2"}
        assert (tmp_path / "blind" / f"{clip}_A.mp4").exists()

    # L'auditeur choisit toujours le clip du système 1.
    resp = tmp_path / "r.csv"
    with open(resp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "choice"])
        for clip, labels in key.items():
            w.writerow([clip, next(k for k, v in labels.items() if v == "system_1")])
    result = blind_test.score(key, [resp])
    assert result["system_1"] == 3 and result["system_2"] == 0


def test_blind_score_rejects_invalid_choice(tmp_path):
    resp = tmp_path / "r.csv"
    resp.write_text("clip,choice\nc1,Z\n")
    with pytest.raises(ValueError):
        blind_test.score({"c1": {"A": "system_1", "B": "system_2"}}, [resp])


def test_blind_prepare_without_common_clips(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    with pytest.raises(ValueError):
        blind_test.prepare(tmp_path / "a", tmp_path / "b", tmp_path / "out")


# --- CLI -------------------------------------------------------------------

def test_cli_wer_and_der(tmp_path, capsys):
    hyp = tmp_path / "h.json"
    hyp.write_text(json.dumps({"segments": [seg(0, 5, "x", "hello world")]}))
    ref = tmp_path / "r.txt"
    ref.write_text("hello there world")
    evaluate.main(["wer", "--ref", str(ref), "--hyp", str(hyp)])
    assert json.loads(capsys.readouterr().out)["wer"] == pytest.approx(1 / 3)
    evaluate.main(["der", "--ref", str(hyp), "--hyp", str(hyp)])
    assert json.loads(capsys.readouterr().out)["der"] == pytest.approx(0, abs=1e-6)


def test_placement_overruns_and_overlap():
    segments = [seg(0, 2), seg(2, 4), seg(4, 6)]
    placed = [{"start": 0, "end": 2.5}, {"start": 2, "end": 3.5}, {"start": 4, "end": 6}]
    r = metrics.placement_overruns(segments, placed)
    assert r["pct_overlapping_next"] == pytest.approx(100 / 3)
    assert r["total_overlap_s"] == pytest.approx(0.5)
    assert r["mean_end_overrun"] == pytest.approx(0.0)  # +0.5 -0.5 +0


def test_placement_overruns_length_mismatch():
    with pytest.raises(ValueError):
        metrics.placement_overruns([seg(0, 1)], [])
