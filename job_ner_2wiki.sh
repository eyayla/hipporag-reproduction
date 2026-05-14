#!/bin/bash
#SBATCH --job-name=ner-2wiki
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/work/coa258/hippoRAG/logs/%j_ner_2wiki.out
#SBATCH --error=/work/coa258/hippoRAG/logs/%j_ner_2wiki.err

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

python main_ner.py \
  --dataset 2wikimultihopqa \
  --llm_base_url "http://192.168.1.209:8000/v1" \
  --llm_name "meta-llama/Llama-3.1-8B-Instruct" \
  --embedding_name nvidia/NV-Embed-v2 \
  --save_dir outputs_ner

echo "=============================="
echo "Bitiş : $(date)"
echo "=============================="
