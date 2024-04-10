Ce document doit servir de guide d'installation minimal pour être en mesure de faire fonctionner l'entraînement des modèles implémentés. Vous pouvez trouver des instructions plus détallées concernant : [ml-agents et unity](https://unity-technologies.github.io/ml-agents/Installation/).

Nous n'**incluons pas** ici l'ensemble des packages pour l'éditeur unity car ils sont trop lourd et ne sont pas indispensables pour tester les algorithmes. Les entrînements/inférences peuvent être lancé à partir des **exécutables** que nous fournissons.

## 1. Cuda

Pour fonctionner, **torch** à besoin que Cuda soit installé sur la machine. Si ce n'est pas déjà fait, vous pouvez installer [Cuda 11.8 ici](https://developer.nvidia.com/cuda-11-8-0-download-archive)

## 2. Python 3.10.12

Il est nécessaire, pour faire fonctionner **ml-agents**, d'avoir un environnement **Python 3.10.12**. Il peut être installé comme ceci avec conda :

```bash
conda create -n ml-agents python=3.10.12
```

Assurez vous d'**activer l'environnement** pour la suite de l'installation.

## 3. Dépendances python

Les dépendances python incluent : **PyTorch**, **ml-agents** et **ml-agents-envs**. Les 2 packages ml-agents release 21 sont inclues ici-même, à la racine du répertoire. Vous pouvez exécter les commandes suivantes pour les installer :

```bash
pip3 install torch -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -e ./ml-agents-envs
pip3 install -e ./ml-agents
```
