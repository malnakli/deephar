#!/bin/bash
#SBATCH --time=00-02:31:20 # "mm", "mm:ss", "hh:mm:ss", "dd-hh", "dd-hh:mm" and "dd-hh:mm:ss".
#SBATCH --account=def-jiayuan
#SBATCH --mail-user=mah.sync24@gmail.com
#SBATCH --mail-type=END
#SBATCH --mem-per-cpu=12000M      # memory; default unit is megabytes
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1              # Number of GPUs (per node)

module load python/3.7
module load scipy-stack
virtualenv --no-download ~/ENVS/pytorch
source  ~/ENVS/pytorch/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
# install pytorch-lightning 
pip install --no-index  https://github.com/williamFalcon/test-tube/archive/0.7.5.zip
pip install --no-index  https://github.com/williamFalcon/pytorch-lightning/archive/master.zip

python -m pytorch.main -d /scratch/malnakli/datasets/MPII/