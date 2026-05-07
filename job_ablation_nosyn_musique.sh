#!/bin/bash
#SBATCH --job-name=ablation-musique
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/work/coa258/hippoRAG/logs/%j_nosyn_musique.out
#SBATCH --error=/work/coa258/hippoRAG/logs/%j_nosyn_musique.err

source ~/.bashrc
conda activate hipporag

export HF_HOME=/scratch/coa258/hf_cache
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY="local"

cd /work/coa258/hippoRAG
mkdir -p logs

echo "=============================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Başlangıç  : $(date)"
echo "=============================="

python main.py \
  --dataset musique \
  --llm_base_url "http://192.168.1.210:8000/v1" \
  --llm_name "meta-llama/Llama-3.1-8B-Instruct" \
  --embedding_name nvidia/NV-Embed-v2 \
  --synonymy_edge_sim_threshold 1.0 \
  --save_dir outputs_nosyn

echo "=============================="
echo "Bitiş : $(date)"
echo "=============================="
