#!/bin/bash

#SBATCH -N 1 # number of nodes
#SBATCH -c 8 # number of cores
#SBATCH -p general # partition
#SBATCH -G a100:1 # GPU's
#SBATCH -t 7-00:00:00 # time in d-hh:mm:ss
#SBATCH -q public # QOS
#SBATCH -o slurm\slurm_files\cnnvtran_mimic_ImageNet22k_swin_100_224_exp01\slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -o slurm\slurm_files\cnnvtran_mimic_ImageNet22k_swin_100_224_exp01\slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Load required modules for job's environment
module load mamba/latest
# Using python, so source activate an appropriate environment
source activate BenchmarkingTransformers
#navigate to project directory
cd /data/jliang12/ayanan/Projects/BenchmarkCNNvTransformer 

python main_classification.py \
	 --data_set MIMIC_CXR \
	 --model swin_base \
	 --init imagenet_21k \
	 --data_dir /data/jliang12/jpang12/dataset/MIMIC_jpeg \
	 --train_list none \
	 --test_list none \
	 --val_list none \
	 --lr 0.1 \
	 --opt sgd \
	 --epochs 100 \
	 --warmup-epochs 0 \
	 --batch_size 64 \
	 --exp_name cnnvtran_mimic_ImageNet22k_swin_100_224_exp01 \
	 --GPU 1 \
	 --workers 8 \
	 --models_dir /scratch/ayanan/Projects/CNNvTransformer/TrainedModels \
	 --resume true