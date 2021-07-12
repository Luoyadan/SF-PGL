#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=SF_PGL
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o tensor_out_visda18_20.txt
#SBATCH -e tensor_err_visda18_20.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

module load anaconda/3.6
source activate /scratch/itee/uqyluo/envs/SF-PGL

srun python train_new.py