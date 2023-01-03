#!/bin/sh
#BSUB -q gpua100
#BSUB -J Classifier
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 03:00
#BSUB -R "rusage[mem=8GB]"
##BSUB -R "select[gpu40gb]" #options gpu40gb or gpu80gb
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err
# -- end of LSF options --

nvidia-smi

source ../envs/3d/bin/activate

python experiments/MedMNIST3D/train_and_eval_pytorch.py --data_flag nodulemnist3d
python experiments/MedMNIST3D/train_and_eval_pytorch.py --data_flag fracturemnist3d
python experiments/MedMNIST3D/train_and_eval_pytorch.py --data_flag adrenalmnist3d
python experiments/MedMNIST3D/train_and_eval_pytorch.py --data_flag vesselmnist3d
python experiments/MedMNIST3D/train_and_eval_pytorch.py --data_flag synapsemnist3d