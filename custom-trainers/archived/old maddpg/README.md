## MADDPG

Ce module est une addaptation de [cette implémentation PettingZoo de MADDPG](https://github.com/Git-123-Hub/maddpg-pettingzoo-pytorch) pour Unity ML-Agents.
L'implémentation utilise le [wrapper PettingZoo](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Python-PettingZoo-API.md) fournit par Unity et fonctionne pour un environnement multi-agents avec un espace d'actions continue.

## Custom Soccer environment

L'environnement SoccerEnv (sur la même branche) à été adapté pour utiliser un espace d'action continue de taille 3 :

1. Déplacement avant/arrière
2. Déplacement latéral
3. Rotation

L'environnement peut être compilé en suivant les instruction de [ml-agents](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Executable.md). Ce n'est pas indispensable.

## Utilisation

### Si vous utiliser directement **l'éditeur unity** :

1. Dans `maddpg/main.py` mettre la valeur de `EXECUABLE_PATH` à `None`.
2. Ouvrir l'environnement `SoccerEnv` dans l'éditeur Unity.
3. Ouvrir la scène `SoccerTwo` dans l'éditeur Unity.
4. Depuis le répertoire `custom-trainers` éxecuter :

```bash
python maddpg\main.py --run_id "<name_of_the_run>"
```

5. Appuyer sur `play` dans l'éditeur Unity.

### Si vous utiliser **l'exécutable** :

1. Dans `maddpg/main.py` mettre la valeur de `EXECUABLE_PATH` au chemin de l'exécutable.
2. Depuis le répertoire `custom-trainers` éxecuter :

```bash
python maddpg\main.py --run_id "<name_of_the_run>"
```

3. Une fenêtre de l'environnement devrait s'ouvrir et l'entrainement devrait commencer.
