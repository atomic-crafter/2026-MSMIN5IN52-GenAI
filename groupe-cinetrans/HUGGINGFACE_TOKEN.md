# Autorisation Hugging Face (obligatoire pour la diarisation)

La détection des locuteurs utilise les modèles **pyannote**, hébergés sur Hugging Face
et protégés ("gated") : il faut un compte, **accepter leurs conditions**, et fournir un token.
Les modèles tournent ensuite **en local** ; le token ne sert qu'au téléchargement.

## 1. Créer un compte
https://huggingface.co/join (gratuit)

## 2. Accepter les conditions des deux modèles
Connecté à votre compte, ouvrez chaque page, remplissez le formulaire (nom, organisation…)
et cliquez sur **Agree and access repository** :

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

> Oublier l'un des deux est la cause n°1 des erreurs `401` / `gated repo` / `Cannot access`.

## 3. Créer le token
1. https://huggingface.co/settings/tokens → **Create new token**
2. Type **Read** (suffisant), nom au choix (ex. `cinetrans`)
3. Copiez-le immédiatement (affiché une seule fois) : il commence par `hf_`

## 4. Le mettre dans le projet
```bash
cp .env.example .env
```
Éditez `.env` :
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```
Pas de guillemets, pas d'espaces. `.env` est ignoré par git : **ne le commitez jamais**
(seul `.env.example`, sans valeur, est versionné).

- **Docker** : `docker-compose.yml` charge `.env` automatiquement.
- **Sans Docker** : `gui.py` lit `.env` ; en ligne de commande, `export HF_TOKEN=hf_...`.

## 5. Vérifier (optionnel)
```bash
pip install huggingface_hub
python -c "from huggingface_hub import model_info; print(model_info('pyannote/speaker-diarization-3.1', token='hf_...').id)"
```
Affiche le nom du modèle si tout est bon ; une erreur 401/403 signifie qu'une étape 2 ou 3 manque.

## Sécurité
- Token compromis (collé dans un chat, un commit, une capture) → le **révoquer** sur la page
  des tokens et en créer un nouveau. Un `git revert` n'efface pas un secret de l'historique.
- Un token `Read` ne permet pas de modifier vos modèles, mais traitez-le comme un mot de passe.
