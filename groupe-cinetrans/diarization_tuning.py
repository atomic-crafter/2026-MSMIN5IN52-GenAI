"""Réglages du clustering pyannote, partagés par extractsrt.py et le banc d'essai de diarisation."""

# Sensibilité de séparation des voix (1 = fusionne plus, 5 = sépare plus) -> seuil de clustering
SPEAKER_THRESHOLDS = {1: 0.85, 2: 0.70, 3: 0.58, 4: 0.46, 5: 0.35}
# pyannote 3.1 fixe min_cluster_size=12 (pensé pour de longs enregistrements) : sur un dialogue
# court, une voix qui parle peu est absorbée par la plus bavarde. On l'abaisse.
MIN_CLUSTER_SIZE = 4


def tune_diarizer(pipe, sensitivity):
    """Ajuste le clustering de pyannote (seuil + taille minimale de cluster)."""
    try:
        params = pipe.parameters(instantiated=True)
        params["clustering"]["threshold"] = SPEAKER_THRESHOLDS[sensitivity]
        params["clustering"]["min_cluster_size"] = MIN_CLUSTER_SIZE
        pipe.instantiate(params)
    except Exception as e:  # ne jamais bloquer le traitement pour un réglage
        print(f"  (réglage de la diarisation ignoré : {e})")
