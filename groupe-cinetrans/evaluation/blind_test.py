"""Écoute à l'aveugle : préférence A/B entre deux systèmes de doublage.

prepare : copie les clips de deux dossiers (même nom de fichier) sous les noms
          <clip>_A / <clip>_B, l'attribution système -> A/B étant tirée au hasard,
          et écrit key.json (à NE PAS montrer aux auditeurs).
score   : lit un ou plusieurs CSV de réponses (clip,choice avec choice ∈ A|B|tie)
          et donne la préférence pour chaque système avec un test du signe exact.
"""
import csv
import json
import random
import shutil
from math import comb
from pathlib import Path


def prepare(dir_a, dir_b, out_dir, seed=None):
    """dir_a / dir_b : les deux systèmes comparés. Retourne la clé {clip: {"A": système, "B": système}}."""
    rng = random.Random(seed)
    dir_a, dir_b, out_dir = Path(dir_a), Path(dir_b), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in dir_a.iterdir() if p.is_file() and (dir_b / p.name).is_file())
    if not names:
        raise ValueError("aucun clip commun aux deux dossiers")
    key = {}
    for name in names:
        clip, ext = Path(name).stem, Path(name).suffix
        systems = [("system_1", dir_a / name), ("system_2", dir_b / name)]
        rng.shuffle(systems)
        key[clip] = {}
        for label, (system, src) in zip("AB", systems):
            shutil.copy(src, out_dir / f"{clip}_{label}{ext}")
            key[clip][label] = system
    # La clé reste hors du dossier distribué aux auditeurs.
    (out_dir.parent / f"{out_dir.name}.key.json").write_text(json.dumps(key, indent=2))
    with open(out_dir / "responses_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "choice"])
        w.writerows([[clip, ""] for clip in key])
    return key


def sign_test_p(wins, losses):
    """p-value bilatérale exacte du test du signe (égalités exclues)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def score(key, response_files):
    """Dépouille les CSV de réponses. `key` : dict ou chemin du key.json."""
    if not isinstance(key, dict):
        key = json.loads(Path(key).read_text())
    wins = {"system_1": 0, "system_2": 0, "tie": 0}
    for path in response_files:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                choice = row["choice"].strip().upper()
                if not choice:
                    continue
                if choice in ("TIE", "="):
                    wins["tie"] += 1
                elif choice in ("A", "B") and row["clip"] in key:
                    wins[key[row["clip"]][choice]] += 1
                else:
                    raise ValueError(f"réponse invalide : {row}")
    total = sum(wins.values())
    return {**wins, "votes": total,
            "p_value": sign_test_p(wins["system_1"], wins["system_2"])}
