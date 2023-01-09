#BSUB -R "span[hosts=1]"

#BSUB -q gpua100
#BSUB -J Classify
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:00

#BSUB -R "rusage[mem=32GB]"

#BSUB -o logs/gpu_%J.out
#BSUB -e logs/gpu_%J.err

nvidia-smi

module load numpy/1.23.3-python-3.10.7-openblas-0.3.21
module load pandas/1.4.4-python-3.10.7

source Karl-Johan/env/bin/activate
pip install -r Karl-Johan/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117

python Karl-Johan/train_insects.py
