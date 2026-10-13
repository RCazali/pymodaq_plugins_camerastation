# Camera Station : état au 2026-10-11 (version 0.2.1)

- **Fait** : extension PyMoDAQ 5.1.10 pour l'ordinateur caméra (B) : analyse à cadence limitée, publication vers A par LECO (compteur `Stats/frame_counter` pour la synchro), enregistrement HDF5, aperçu, caméra simulée.
- **Scripts d'analyse** rechargés à chaud depuis `~/.pymodaq/camera_station/scripts` (pas de redémarrage ; envoyés à A seulement si « Publish to A »).
- **Corrigé le 11/10** : une image de bruit seul (laser éteint) donnait `valid = 1` ; elle donne maintenant NaN et `valid = 0` sans réglage.
- **Tests** : 59 réussis au bureau (caméras simulées, vrais viewers PyMoDAQ). **Rien n'a encore tourné au labo.**
- **Limite connue** : si la fluorescence du disque a changé depuis la référence (pompe), la caméra disque voit encore un signal laser éteint : régler *Min signal* ou reprendre une référence.
- **Sans dossier partagé A/B** : les images déclenchées par les événements de A restent désactivées.
- **Prochaine étape** : essai sur l'ordinateur caméra avec les vraies caméras (voir « Commissioning » et « Points to check » dans `README.md`).
