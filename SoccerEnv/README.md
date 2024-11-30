# Environnement SoccerTwo personnalisé

## Caracteristiques

- **Observation** : 16 dimensions (8 \* Vec2), Continues, Observabilité totale

  1. Position du joueur (x, z)
  2. Vélocité du joueur (x, z)
  3. Position du coéquipier (x, z)
  4. Position des adversaires 2\*(x, z)
  5. Position de la balle (x, z)
  6. Vélocité de la balle (x, z)
  7. Position du but de l'équipe (x, z)
  8. Position du but adverse (x, z)

Les observations sont relatives au repère du joueur.

- **Action** : Continue (3 dimensions), Déterministe, normalisés entre -1 et 1
  - Déplacement sur x
  - Déplacement sur z
  - Rotation sur y

## Installation

1. Ajouter le dossier `SoccerEnv` comme nouveau projet dans Unity
2. (ré)Installer les packages `com.unity.ml-agents` et `com.unity.ml-agents.extensions` dans le projet Unity

## Troubleshooting

### 1. System.Reflection.TargetInvocationException: Exception has been thrown by the target of an invocation.

Sur le menu superieur de la fenêtre Unity, désactiver la compilation du _Burst_ : `Jobs` -> `Burst` -> `Enable Compilation`
