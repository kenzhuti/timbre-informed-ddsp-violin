#!/bin/bash
#SBATCH --job-name=uniSynth
#SBATCH -n 16
#SBATCH --mem 30000
#SBATCH -p high                     # Partition to submit to
#SBATCH -o %x-%j.out # File to which STDOUT will be written
#SBATCH -e %x-%j.err # File to which STDERR will be written

module load libsndfile
module load RubberBand
module load FFmpeg
module load Miniconda3
eval "$(conda shell.bash hook)"
conda init bash
conda activate supervised


python label_dataset.py -p "/homedtic/ntamer/musical-etudes/clarinet-etudes"

