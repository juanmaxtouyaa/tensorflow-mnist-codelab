#!/bin/bash
#SBATCH -J mnist-notebooks
#SBATCH -p short
#SBATCH --gres=gpu:a100:1
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out

# Exécute les notebooks du codelab sur un GPU A100 de FinisTerrae III et
# enregistre toutes les sorties dans les fichiers .ipynb.
#
#   sbatch run_notebooks.sbatch                        # les 5 notebooks
#   sbatch run_notebooks.sbatch keras_01_mnist.ipynb   # un seul notebook

module load cesga/system
module load tensorflow/2.10.1-gpu-conda
source "$STORE/envs/mnist-tf210/bin/activate"

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -gt 0 ]; then
    NOTEBOOKS=("$@")
else
    NOTEBOOKS=(keras_0*.ipynb)
fi

echo "Job $SLURM_JOB_ID sur $(hostname), $(date)"
nvidia-smi
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('TensorFlow', tf.__version__, gpus)
assert gpus, 'TensorFlow ne voit aucun GPU'
"

total_start=$(date +%s)
for nb in "${NOTEBOOKS[@]}"; do
    start=$(date +%s)
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.kernel_name=python3 "$nb"
    echo "TEMPS $nb $(( $(date +%s) - start )) s"
done
echo "TEMPS TOTAL $(( $(date +%s) - total_start )) s"
