# HippoRAG 2 Reproduction

Reproduction of **HippoRAG 2: From RAG to Memory** (ICML 2025) on UTSA HPC.

- Original paper: [arXiv 2502.14802](https://arxiv.org/abs/2502.14802)
- Original repo: [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)
- Results comparison: [results.md](results.md)

## Setup

### 1. Environment

```bash
conda create -n hipporag python=3.10 -y
conda activate hipporag
pip install hipporag==2.0.0a3
pip install openai==1.58.1 pydantic==2.10.4 tiktoken==0.7.0 tokenizers==0.20.0
pip install litellm==1.51.0 --no-deps
pip install boto3 python-igraph==0.11.8 einops tenacity==8.5.0 gritlm==1.0.2
pip install vllm==0.6.6.post1
```

### 2. HuggingFace Login

```bash
huggingface-cli login
```

### 3. Download Model

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.1-8B-Instruct', cache_dir='/path/to/hf_cache')
"
```

## Running on SLURM HPC

### Step 1: Start vLLM Server

```bash
sbatch job_vllm_server.sh
# Get node IP
srun --jobid=<JOBID> hostname -i
```

### Step 2: Update IP in Scripts

```bash
for script in job_hipporag_musique.sh job_hipporag_hotpotqa.sh job_hipporag_2wiki.sh job_ircot_musique.sh; do
    sed -i "s|http://.*:8000/v1|http://<VLLM_IP>:8000/v1|g" $script
done
```

### Step 3: Run HippoRAG

```bash
sbatch --exclude=<VLLM_NODE> job_hipporag_musique.sh
sbatch --exclude=<VLLM_NODE> job_hipporag_hotpotqa.sh
sbatch --exclude=<VLLM_NODE> job_hipporag_2wiki.sh
```

### Step 4: Run IRCoT (Proposed Improvement)

```bash
sbatch --exclude=<VLLM_NODE> job_ircot_musique.sh
```

## Reproduce All Tables

```bash
python reproduce_tables.py
```

## Results Summary

| Dataset | R@2 (Paper) | R@2 (Ours) | F1 (Ours) |
|---------|------------|------------|-----------|
| MuSiQue | 41.0 | **52.0** | 0.384 |
| 2Wiki | 71.5 | **73.4** | 0.595 |
| HotpotQA | 59.0 | **78.3** | 0.686 |

## Proposed Improvement: IRCoT Integration

IRCoT was available in HippoRAG 1 but removed in HippoRAG 2. We re-implemented it in `main_ircot.py`. See [results.md](results.md) for details.

## Improvement Experiments

We implemented and evaluated four improvements on HippoRAG 2:

### 1. IRCoT Integration
IRCoT was available in HippoRAG 1 but removed in HippoRAG 2. We re-implemented it in `main_ircot.py`.

**Hypothesis:** Iterative retrieval with chain-of-thought reasoning improves multi-hop question performance.

**Result:** No improvement — HippoRAG 2 already achieves high single-step recall, confirming the paper's efficiency claims.

```bash
sbatch job_ircot_musique.sh
sbatch job_ircot_hotpotqa.sh
sbatch job_ircot_2wiki.sh
```

### 2. Better NER (Key Concepts Extraction)
Modified the NER prompt to extract both named entities AND key concepts (relations, attributes).

**Hypothesis:** Extracting relational concepts alongside entities will activate more relevant graph nodes during PPR.

**Result:** Marginal improvement on MuSiQue (+0.2 R@2) and HotpotQA (+0.1 EM). Negligible effect on 2Wiki.

```bash
sbatch job_ner_musique.sh
sbatch job_ner_hotpotqa.sh
sbatch job_ner_2wiki.sh
```

### 3. Query Expansion
Used LLM to expand queries with related concepts before retrieval (`main_qe.py`).

**Result:** Inconsistent — improved HotpotQA R@2 (+2.7) but hurt MuSiQue and 2Wiki.

### 4. Entity Linking top-k=10
Increased entity linking top-k from 5 to 10 to activate more graph nodes (`main_linking.py`).

**Result:** Improved HotpotQA R@2 significantly (+5.7) but hurt 2Wiki.

## Overall Conclusion

HippoRAG 2 is already highly optimized. Simple improvements provide marginal or inconsistent gains. The paper's core claim — that single-step retrieval matches iterative methods — holds for HippoRAG 2 as well.
