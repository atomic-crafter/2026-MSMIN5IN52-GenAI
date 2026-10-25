# CineTrans — doublage multi-locuteurs

## Groupe

| Membre | Contribution principale |
|---|---|
| @atomic-crafter | Transcription audio → SRT (WhisperX, diarisation, émotions), interface web, intégration de toute la chaîne |
| @Daphneej91 | SRT → audio : synthèse vocale (Kokoro) et resynchronisation |
| @PaulineChallenge | Traduction des segments SRT (LLM) |

## Sujet

- **Identifiant du catalogue :** D2 — Doublage multi-locuteurs mesuré
- **Problématique :** doubler un extrait vidéo en conservant la distinction des locuteurs et le calage temporel.
- **MVP :** vidéo → transcription avec locuteurs et émotions → traduction → voix synthétiques par locuteur → vidéo doublée, pilotable depuis une interface web.

## Architecture

```
vidéo ──► extractsrt.py ──► JSON/SRT (segments, locuteurs, émotions)
              (WhisperX + pyannote)
      ──► translate_json.py ──► JSON traduit (LLM compatible OpenAI)
      ──► dubbing.py ──► voix Kokoro par locuteur, calage temporel, vidéo doublée (ffmpeg)
app.py : interface web Flask reliant les trois étapes
```

Les modèles pyannote sont protégés : voir [HUGGINGFACE_TOKEN.md](HUGGINGFACE_TOKEN.md).

## Installation

```bash
cp .env.example .env   # renseigner HF_TOKEN (non commité), et les variables de traduction
docker compose build
```

## Utilisation

```bash
docker compose up --build        # interface web : http://localhost:8000
docker compose --profile cli run --rm transcribe
docker compose --profile cli run --rm translate
docker compose --profile cli run --rm dubbing
```

Placer une vidéo libre de droits dans `videos/` (dossier ignoré par git) ou l'envoyer via l'interface web.

## Tests

```bash
pip install pytest numpy soundfile
python -m pytest tests
```

Les tests couvrent les métriques d'évaluation (WER, erreur de diarisation, décalage temporel, écoute à l'aveugle) sur des données synthétiques, y compris les cas d'échec (référence vide, réponse invalide, aucun clip commun, segment muet). Ils ne lancent ni Whisper ni Kokoro.

## Résultats et évaluation

Métriques du catalogue D2, calculées par `evaluate.py` :

```bash
python evaluate.py wer --ref reference.txt --hyp segments.json          # WER
python evaluate.py der --ref reference.rttm --hyp segments.json         # erreur de diarisation
python evaluate.py timing --segments segment_fr.json --audio fr.wav --placed fr.placed.json   # décalage temporel
python evaluate.py blind-prepare --system-1 clips_ours --system-2 clips_baseline --out blind
python evaluate.py blind-score --key blind.key.json --responses r1.csv r2.csv   # écoute à l'aveugle
```

### Baseline

`python dubbing.py segments.json --baseline` : une seule voix pour tous les locuteurs et aucun ajustement de durée (la réplique est posée à son début, quelle que soit sa longueur).

### Résultats mesurés (1 extrait)

Extrait de film anglais de 76 s (41 segments), traduit en français, **un seul extrait et un seul run** : ces chiffres sont indicatifs, pas statistiques. L'extrait est un film commercial utilisé en local uniquement ; il n'est pas versionné.

| Mesure | Avant correction | Pipeline actuel | Baseline |
|---|---|---|---|
| Décalage moyen du début de réplique (s, valeur absolue) | 0,31 | **0,06** | 0,05 |
| Répliques qui débordent de plus de 0,2 s | 46 % | 27 % | 39 % |
| Répliques qui chevauchent la suivante | — | **0 %** | 24 % (2,1 s au total) |
| WER aller-retour (Whisper `small` sur la piste doublée vs texte traduit) | — | 0,19 | **0,11** |

- *Avant correction* : la mesure a révélé que Kokoro ajoute ~0,37 s de silence en tête de chaque réplique, soit un retard systématique ; `dubbing.py` supprime désormais ce silence.
- Le WER aller-retour mesure l'intelligibilité de la voix de synthèse, pas la qualité de la transcription : la référence est le texte traduit, pas une transcription humaine. Le pipeline actuel est **moins intelligible** que la baseline : accélérer et tronquer les répliques pour tenir dans le temps coupe des mots. Compromis à discuter.
- **Erreur de diarisation (DER)**, mesurée sur 4 fichiers de VoxConverse (2 à 4 locuteurs, 106 à 260 s, annotations CC BY 4.0, audio non versionné) avec `python -m evaluation.bench_diarization` (pyannote 3.1, nombre de locuteurs inconnu / connu) :

  | Réglage | DER moyen (inconnu) | DER moyen (nb. connu) |
  |---|---|---|
  | pyannote par défaut | 0,087 | 0,085 |
  | réglage projet, sensibilité 2 (défaut de l'interface) | **0,075** | **0,063** |
  | réglage projet, sensibilité 4 | 0,151 | 0,123 |

  La sensibilité 4 sépare trop (jusqu'à 9 locuteurs détectés pour 3 réels) : à éviter. Sur un seul fichier (`sldwj`), `min_cluster_size=4` fait passer le DER de 0,154 à 0,053 ; sur les trois autres il ne change rien.
- **Pourquoi l'extrait de film n'a qu'un seul locuteur :** ce n'est pas un problème de SRT (le SRT reprend l'étiquette donnée par pyannote). Avec sensibilité 4 et 2 locuteurs demandés, pyannote attribue 51 s à un locuteur et 2 s à l'autre. Avec les autres réglages testés, les étiquettes se répartissent en blocs temporels (7 répliques de `0`, puis 13 de `1`) alors que la scène est un échange rapide : pyannote ne sépare pas ces voix (voix de jeunes femmes proches, mélange film avec musique et bruitages). Aucun réglage testé n'est validé sur cet extrait, faute d'annotation de référence. Pour démontrer le doublage multi-voix, utiliser un enregistrement aux voix plus distinctes (entretien, podcast).
- **Écoute à l'aveugle : non réalisée** (le dépouillement est prêt, il manque les auditeurs).

## Limites et plan B

À compléter. Plan B prévu par le catalogue : TTS local Kokoro et extrait libre.

## Répartition du travail

Voir le tableau des membres. Antonin a réalisé la partie audio → SRT et l'interface web, puis fusionné les trois parties ; Daphnée a réalisé SRT → audio ; Pauline a réalisé la traduction des SRT.

Des parties du code ont été écrites avec l'aide d'un agent de code (Claude Code).

## Licence et références

Code : à préciser. Modèles : WhisperX / Whisper, pyannote.audio (conditions d'accès Hugging Face), Kokoro. Aucun média commercial n'est versionné.
