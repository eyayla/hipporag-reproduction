#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=/work/coa258/hippoRAG/logs/%j_vllm.out
#SBATCH --error=/work/coa258/hippoRAG/logs/%j_vllm.err

source ~/.bashrc
conda activate hipporag

export HF_HOME=/scratch/coa258/hf_cache
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token)
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Node: $SLURMD_NODENAME"
echo "Başlangıç: $(date)"

vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --dtype=half \
  --port 8000
