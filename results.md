# Reproduction Results

This document compares our reproduced results with the original HippoRAG paper.

**Note:** The original paper (NeurIPS 2024) used HippoRAG 1. We reproduced using HippoRAG 2 (ICML 2025). HippoRAG 2 uses an improved graph construction method, which explains differences in graph statistics (Table 1) and improved retrieval performance (Table 2).

**Setup:**
- LLM: `meta-llama/Llama-3.1-8B-Instruct` (via vLLM)
- Embedding: `nvidia/NV-Embed-v2`
- Hardware: UTSA HPC (Tesla V100S-PCIE-32GB)

---

## Table 1: Dataset and Knowledge Graph Statistics

| Metric | Paper MuSiQue | Ours MuSiQue | Paper 2Wiki | Ours 2Wiki | Paper HotpotQA | Ours HotpotQA |
|--------|--------------|--------------|-------------|------------|----------------|---------------|
| # of Passages | 11,656 | 11,656 ✅ | 6,119 | 6,119 ✅ | 9,221 | 9,811 ⚠️ |
| # of Unique Nodes | 91,729 | 90,139 ✅ | 42,694 | 46,383 ⚠️ | 82,157 | 86,348 ⚠️ |
| # of Unique Edges | 21,714 | 749,686 ❌ | 7,867 | 362,211 ❌ | 17,523 | 643,299 ❌ |

**Note:** Edge count difference is due to HippoRAG 2 using a different graph construction method than HippoRAG 1.

---

## Table 2: Single-step Retrieval Performance

| Method | MuSiQue R@2 | MuSiQue R@5 | 2Wiki R@2 | 2Wiki R@5 | HotpotQA R@2 | HotpotQA R@5 |
|--------|------------|------------|----------|----------|-------------|-------------|
| HippoRAG (Paper, Contriever) | 41.0 | 52.1 | 71.5 | 89.5 | 59.0 | 76.2 |
| **HippoRAG 2 (Ours)** | **52.0** | **73.2** | **73.4** | **88.7** | **78.3** | **95.3** |

Our results outperform the original paper, consistent with HippoRAG 2 improvements.

---

## QA Performance

| Dataset | ExactMatch | F1 |
|---------|-----------|-----|
| MuSiQue | 0.290 | 0.384 |
| HotpotQA | 0.563 | 0.686 |
| 2WikiMultiHopQA | 0.525 | 0.595 |

---

## Table 3: Multi-step Retrieval Performance (IRCoT + HippoRAG)

| Method | MuSiQue R@2 | MuSiQue R@5 | 2Wiki R@2 | 2Wiki R@5 | HotpotQA R@2 | HotpotQA R@5 |
|--------|------------|------------|----------|----------|-------------|-------------|
| IRCoT + HippoRAG (Paper) | 43.9 | 56.6 | 75.3 | 93.4 | 65.8 | 82.3 |
| **IRCoT + HippoRAG 2 (Ours)** | TBD | TBD | TBD | TBD | TBD | TBD |

---

## Proposed Improvement: IRCoT Integration for HippoRAG 2

**Idea:** IRCoT was available in HippoRAG 1 but not carried over to HippoRAG 2. We re-implemented it in `main_ircot.py`.

**Hypothesis:** Iterative retrieval with chain-of-thought reasoning improves performance on complex multi-hop questions.

**Implementation:** See `main_ircot.py` for the full implementation.
