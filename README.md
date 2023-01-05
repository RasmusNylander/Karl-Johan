# Karl-Johan 🍄

## Install

`pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116`

## Run Jupyter notebook on HPC

- run on HPC `jupyter notebook --no-browser --port=40000 --ip=$HOSTNAME`
- run on local computer `ssh -i path/to/sshkey USERNAME@l1.hpc.dtu.dk -g -L8080:HOSTNAME:PORT -N`

## Problem description

## Meeting plan
- Thursday 5th. at 15
- Monday 9th. at 14
- Thursday 12th. at 14

## Initial plan
- Phase 1
  - Get familiar with Monai by classifing MedNIST data using deep learning. Comparison between two or more deep learning methods for 3D classification of the six 3D datasets. 
  - Investigate tools network visualisation e.g. GradCAM
  - Apply the network visualisation to the MedNIST models
  
- Phase 2
  - Structure the insect data for deep learning-based 3D classification (sort, scale, etc.).
  - Implement the same deep learning-based classification models for the insect dataset as used for the MedNIST
  - Implement one or more improved deep learning models for the insect dataset
  - Verify the data by applying network visualisation to insect model
 
## TODOs
- [x] Create requirement file (so all our envs are the same)
- [x] Make sure all have acess to HPC and know how to use it
- [ ] Create a problem description

## Questions for superviser meeting:
- Is hand-in only a slide deck with comments, and how elaborate should the comments be?
- Since there is only 500 images, should we do some image augmentation, and which methods would be a good idea, rotating, scaling, bluring?
- Is it relevant to train models for all 6 3D MedNIST datasets?
- Is it nessesary to use the monai models? 
   
