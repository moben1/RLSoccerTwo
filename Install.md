Résumé et suppléments pour le [guide d'installation ml-agents](https://unity-technologies.github.io/ml-agents/Installation/)

## 0. Avant de commencer

- Installer [Cuda 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) (nécessaire pour torch).
  Rien de spécial, l'installation express (par default) fonctionne bien.

- Créer un environnement python 3.10.12 (comme recommandé)

  Je n'ai pas eu de problème (j'utilise conda), mais notez tout quand même que :

  > If you are using Windows, please install the x86-64 version and not x86

## 1. Cloner le repo

```bash
git clone --branch release_21 https://github.com/Unity-Technologies/ml-agents.git
```

## 2. Changer la version de numpy

Il semble qu'il y ai un conflit avec la version requise de numpy lors de l'installation, notemment sur Windows. Cependant la solution que j'ai trouvé ne semble pas très "clean", je vous suggère d'essayer de passer cette étape, et de l'effectuer en cas de probème par la suite.

Si toutefois vous rencontrez cette erreur lors de l'installation de ml-agents (**Failed building wheel for numpy**),

```bash
  Building wheel for numpy (pyproject.toml) ... error
  error: subprocess-exited-with-error

  × Building wheel for numpy (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [271 lines of output]
      setup.py:63: RuntimeWarning: NumPy 1.21.2 may not yet support Python 3.10.
[....]
[....]
      TypeError: CCompiler_spawn() got an unexpected keyword argument 'env'

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for numpy
```

voici comment la contourner :

https://github.com/Unity-Technologies/ml-agents/issues/5826#issuecomment-1403248637

## 3. Installer les dépendances python

"Advanced: Local Installation for Development" au cas où on voudrais apporter des modifications.

- **PyTorch** 1.13.1 :

  ```bash
  pip3 install torch -f https://download.pytorch.org/whl/torch_stable.html
  ```

- **ml-agents** :

  Les deux dossiers **ml-agents-envs** et **ml-agents** sont dans le dossier cloné :

  ```bash
  cd path/to/ml-agents
  pip3 install -e ./ml-agents-envs
  pip3 install -e ./ml-agents
  ```

  en cas d'erreur, aller à l'étape 2.

A ce stade vous devriez pouvoir exécuter la commande suivante dans l'environnement python :

```bash
mlagents-learn --help
```

## 4. Installer Unity et les package Unity

- Télécharger Unity Hub et Unity 2020.3.4f1 (LTS). C'est dans cette version que les exemples d'environnements ont été implémentés.

- Vous ensuite pouvez ouvrire le projet avec les exemples dans Unity Hub :

  - Menu `"Projects" > "Add"`

  - Sélectionner ce dossier `path\to\ml-agents\Project`

- Installer les packages Unity dans un nouveau projet :

  Menu `"Window" > "Package Manager" > "+" en haut a gauche > "Add package from disk..."`

  1. **com.unity.ml-agents** : `path\to\ml-agents\com.unity.ml-agents\package.json`

  2. **com.unity.ml-agents.extensions** : `path\to\ml-agents\com.unity.ml-agents.extensions\package.json`

## 5. [Exécuter les exemples](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Getting-Started.md)

- Ouvrir le dossier "Projets" dans Unity
- Choisir un exemple et ouvrir la scène
- Dans `path\to\ml-agents\config` trouver le fichier de configuration correspondant à l'environnement
- Dans un terminal (dans l'environnement python) executer :

  ```bash
  mlagents-learn "...\ml-agents\config\..." --run-id="run_name"
  ```

- Dans Unity, appuyer sur play.

  Normalement des logs apparaissent régulièrement avec la progression de l'apprentissage. La `mean reward` devrait augmenter
