# Codelab TensorFlow/Keras MNIST sur FinisTerrae III

Notebooks du codelab Google [TensorFlow, Keras et deep learning, sans doctorat](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist?hl=fr)
(dossier `tensorflow-mnist-tutorial` de
[GoogleCloudPlatform/tensorflow-without-a-phd](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd)).

L'étape 14 du codelab propose d'entraîner le modèle dans le cloud sur du matériel
puissant. Ici, ce rôle est joué par un GPU NVIDIA A100 du supercalculateur
FinisTerrae III (CESGA). Tous les notebooks ont été exécutés sur ce GPU et
**toutes les sorties sont enregistrées dans les fichiers `.ipynb`** : il suffit de les
ouvrir sur GitHub, sans rien relancer.

## Résultats

Exécution du 4 octobre 2026 (job Slurm 10308194), 10 époques par notebook.

| Notebook | Étapes du codelab | Modèle | Précision de validation finale | Attendu (codelab) | Temps d'exécution |
| --- | --- | --- | ---: | ---: | ---: |
| [`keras_01_mnist.ipynb`](keras_01_mnist.ipynb) | 3 et 5 | Une couche dense softmax, SGD | **89,69 %** | ~90 % | 33 s |
| [`keras_02_mnist_dense.ipynb`](keras_02_mnist_dense.ipynb) | 6 et 7 | Couches denses ReLU, Adam | **97,84 %** | ~97 % | 25 s |
| [`keras_03_mnist_dense_lrdecay_dropout.ipynb`](keras_03_mnist_dense_lrdecay_dropout.ipynb) | 8 et 9 | Décroissance du taux d'apprentissage, dropout 25 % | **97,87 %** | ~98 % | 26 s |
| [`keras_04_mnist_convolutional.ipynb`](keras_04_mnist_convolutional.ipynb) | 11 et 12 | Trois couches convolutives, dropout 40 % | **99,05 %** | > 99 % | 59 s |
| [`keras_05_mnist_batch_norm.ipynb`](keras_05_mnist_batch_norm.ipynb) | 13 | Convolutions + batch normalization | **99,49 %** | ~99,5 % | 55 s |

- **Temps total** : 198 s pour les cinq notebooks (3 min 34 s pour le job Slurm complet).
- La précision de validation est celle de la dernière époque (`val_accuracy`),
  calculée sur les 10 000 images du jeu de test MNIST, que le codelab appelle
  « validation ».
- Les résultats varient légèrement d'une exécution à l'autre (initialisation
  aléatoire, mélange des données, dropout).
- Les étapes 1–2, 4, 10 et 15 du codelab sont des étapes d'introduction, de théorie
  ou de conclusion, sans notebook propre.

## Environnement d'exécution

| | |
| --- | --- |
| Machine | FinisTerrae III (CESGA), nœud `a100-29`, partition `short` |
| GPU (`nvidia-smi`) | 1 × NVIDIA A100-PCIE-40GB, pilote 570.86.15 |
| Ressources Slurm | `--gres=gpu:a100:1 -c 32 --mem=64G -t 00:45:00` |
| TensorFlow | 2.10.1 (module CESGA `tensorflow/2.10.1-gpu-conda`, CUDA 11.2, Keras 2.10) |
| Python | 3.10.15 |
| Paquets ajoutés | [`requirements.txt`](requirements.txt) : matplotlib, Pillow, ipykernel, nbconvert |

TensorFlow 2.10 a été choisi car c'est une version fournie par le CESGA pour les
A100, et Keras 2 est le plus proche de la version d'origine du codelab
(TensorFlow 2.2) : le code n'a presque pas besoin d'être modifié.

## Modifications par rapport au codelab

Le code et la structure des notebooks sont ceux du codelab. Chaque modification est
signalée dans le notebook par une note « **Modification** » juste avant la cellule
concernée :

- **Données** : le bucket `gs://mnist-public` n'est plus accessible publiquement
  (erreur 403). Les mêmes fichiers MNIST au format IDX sont téléchargés depuis le
  miroir CVDF de Google par [`scripts/download_mnist.py`](scripts/download_mnist.py)
  (avec vérification SHA-256) dans `data/mnist/` (non versionné). Seuls les
  chemins `gs://mnist-public/` → `data/mnist/` ont changé.
- **Compatibilité** avec les versions récentes de NumPy, Matplotlib et Keras :
  `np.float` → `float`, `plt.grid(b=None)` → `plt.grid(False)`,
  `Adam(lr=0.01)` → `Adam(learning_rate=0.01)`.
- **Exercice du codelab** : `Dropout(0.4)` ajouté au réseau convolutif (notebook 04).

Remarques sur les sorties :

- Seule la ligne de la dernière époque reste affichée : la fonction de rappel
  `PlotTraining` du codelab efface la sortie à chaque époque pour redessiner les
  courbes d'entraînement, qui montrent, elles, toutes les époques.
- Les messages de TensorFlow sur fond rouge (oneDNN, `cuBLAS factory`,
  `cache_dataset_ops`) sont des avertissements sans effet sur les résultats. On y
  voit aussi la ligne `Created device ... NVIDIA A100-PCIE-40GB`, qui confirme que
  l'entraînement s'est fait sur le GPU.

## Relancer l'exécution

Préparation, une seule fois, sur un nœud de connexion de FinisTerrae III (accès à
Internet) :

```bash
module load cesga/system
module load tensorflow/2.10.1-gpu-conda
python -m venv --system-site-packages "$STORE/envs/mnist-tf210"
source "$STORE/envs/mnist-tf210/bin/activate"
pip install -r requirements.txt
python scripts/download_mnist.py
```

Ensuite, depuis le dossier du dépôt, une seule commande exécute les cinq notebooks
sur un A100 et enregistre les sorties dans les fichiers `.ipynb` :

```bash
sbatch run_notebooks.sbatch
```

Le journal du job est écrit dans `logs/`. Pour un seul notebook :
`sbatch run_notebooks.sbatch keras_01_mnist.ipynb`.

## Licence

Les notebooks conservent leur licence d'origine (Apache 2.0, Google LLC).
