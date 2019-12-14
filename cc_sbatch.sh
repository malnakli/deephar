#!/bin/bash
#SBATCH --time=00-01:01:20 # "mm", "mm:ss", "hh:mm:ss", "dd-hh", "dd-hh:mm" and "dd-hh:mm:ss".
#SBATCH --account=def-jiayuan
#SBATCH --mail-user=mah.sync24@gmail.com
#SBATCH --mail-type=END
#SBATCH --mem-per-cpu=8024M      # memory; default unit is megabytes
#SBATCH --cpus-per-task=4
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

# wget https://files.pythonhosted.org/packages/86/45/a273fe3f8fe931a11da34fba1cb74013cfc70dcf93e5d8d329c951dc44c5/Keras-2.1.4-py2.py3-none-any.whl
pip install Keras-2.1.4-py2.py3-none-any.whl 

python exp/pennaction/eval_penn_ar_pe_merge.py output/eval-penn
