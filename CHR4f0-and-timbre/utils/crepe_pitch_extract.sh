#!/bin/bash
#SBATCH --job-name=pe
#SBATCH -n 9
#SBATCH --gres=gpu:1
# #SBATCH --mem 10000
#SBATCH -p high                     # Partition to submit to
#SBATCH -o %x-%j.out # File to which STDOUT will be written
#SBATCH -e %x-%j.err # File to which STDERR will be written

module --ignore-cache load "libsndfile"
#module load TensorFlow/2.7.1-foss-2020b-CUDA-11.4.3
module --ignore-cache load "FFmpeg/4.3.2-GCCcore-10.2.0"
module --ignore-cache load "Anaconda3"
module --ignore-cache load "PyTorch/1.10.0-GCCcore-10.2.0-CUDA-11.4.3"
module --ignore-cache load "CUDA"
module --ignore-cache load "cuDNN"

#eval "$(conda shell.bash hook)"
#conda init bash
#conda create -n torch110
#conda activate torch110

#pip install mir_eval
#eval "$(conda shell.bash hook)"
#conda init bash
#conda activate supervised

#pip install libf0
#pip install crepe
python extract_pitch_crepe.py
