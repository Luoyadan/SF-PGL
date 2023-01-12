#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=SF_PGL
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o logs/ablation_out_visda17_res_batch4_EF5_ep2_the0.7_eta0.01.txt
#SBATCH -e logs/ablation_err_visda17_res_batch4_EF5_ep2_the0.7_eta0.01.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

module load anaconda/3.6
source activate /scratch/itee/uqyluo/envs/SF-PGL

srun python visda_train.py --entropy_loss=0.01