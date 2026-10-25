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
python evaluate.py timing --segments segment_fr.json --audio outputs/fr.wav   # décalage temporel
python evaluate.py blind-prepare --system-1 clips_ours --system-2 clips_baseline --out blind
python evaluate.py blind-score --key blind.key.json --responses r1.csv r2.csv   # écoute à l'aveugle
```

Résultats mesurés face à la baseline : à compléter (aucune mesure réalisée pour l'instant).

## Limites et plan B

À compléter. Plan B prévu par le catalogue : TTS local Kokoro et extrait libre.

## Répartition du travail

Voir le tableau des membres. Antonin a réalisé la partie audio → SRT et l'interface web, puis fusionné les trois parties ; Daphnée a réalisé SRT → audio ; Pauline a réalisé la traduction des SRT.

Des parties du code ont été écrites avec l'aide d'un agent de code (Claude Code).

## Licence et références

Code : à préciser. Modèles : WhisperX / Whisper, pyannote.audio (conditions d'accès Hugging Face), Kokoro. Aucun média commercial n'est versionné.
