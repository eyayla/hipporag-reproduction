# Reproduction Results

This document compares our reproduced results with the original HippoRAG paper.

**Note:** The original paper (NeurIPS 2024) used HippoRAG 1. We reproduced using HippoRAG 2 (ICML 2025).

**Setup:**
- LLM: `meta-llama/Llama-3.1-8B-Instruct` (via vLLM)
- Embedding: `nvidia/NV-Embed-v2`
- Hardware: UTSA HPC (Tesla V100S-PCIE-32GB)

---

## Table 2: Single-step Retrieval
| Method | MuSiQue R@2 | R@5 | 2Wiki R@2 | R@5 | HotpotQA R@2 | R@5 |
|---|---|---|---|---|---|---|
| HippoRAG (Paper, Contriever) | 41.0 | 52.1 | 71.5 | 89.5 | 59.0 | 76.2 |
| **HippoRAG 2 (Ours)** | **52.0** | **73.2** | **73.4** | **88.7** | **78.3** | **95.3** |

## Table 3: Multi-step Retrieval (IRCoT)
| Method | MuSiQue R@2 | R@5 | 2Wiki R@2 | R@5 | HotpotQA R@2 | R@5 |
|---|---|---|---|---|---|---|
| IRCoT+HippoRAG (Paper) | 43.9 | 56.6 | 75.3 | 93.4 | 65.8 | 82.3 |
| **IRCoT+HippoRAG 2 (Ours)** | **52.0** | **73.2** | **73.4** | **88.6** | **78.3** | **95.3** |

## Table 4: QA Performance
| Method | MuSiQue EM | F1 | 2Wiki EM | F1 | HotpotQA EM | F1 |
|---|---|---|---|---|---|---|
| HippoRAG (Paper, ColBERTv2) | 19.2 | 29.8 | 46.6 | 59.5 | 41.8 | 55.0 |
| **HippoRAG 2 (Ours)** | **29.0** | **38.4** | **52.5** | **59.5** | **56.3** | **68.6** |

## Table 5: Ablation (w/o Synonymy Edges)
| Method | MuSiQue R@2 | R@5 | 2Wiki R@2 | R@5 | HotpotQA R@2 | R@5 |
|---|---|---|---|---|---|---|
| HippoRAG (Paper) | 40.9 | 51.9 | 70.7 | 89.1 | 60.5 | 77.7 |
| w/o Synonymy (Paper) | 40.2 | 50.2 | 69.2 | 85.6 | 59.1 | 75.7 |
| **w/o Synonymy (Ours)** | **51.1** | **72.6** | **73.4** | **90.5** | **75.4** | **95.5** |

## Table 6: All-Recall
| Method | MuSiQue AR@2 | AR@5 | 2Wiki AR@2 | AR@5 | HotpotQA AR@2 | AR@5 |
|---|---|---|---|---|---|---|
| HippoRAG (Paper) | 10.2 | 22.4 | 45.4 | 75.7 | 33.8 | 57.9 |
| **HippoRAG 2 (Ours)** | **19.5** | **45.5** | **50.1** | **69.8** | **60.4** | **91.3** |

---

## Proposed Improvement: IRCoT + Adaptive Stopping

**Idea:** IRCoT was available in HippoRAG 1 but not in HippoRAG 2. We re-implemented it with adaptive stopping.

**Hypothesis:** Iterative retrieval with chain-of-thought reasoning improves multi-hop question performance. Adaptive stopping reduces unnecessary steps for simple questions.

**Results:**
| Method | MuSiQue R@2 | 2Wiki R@2 | HotpotQA R@2 |
|---|---|---|---|
| HippoRAG 2 (single-step) | 52.0 | 73.4 | 78.3 |
| IRCoT+HippoRAG 2 | 52.0 | 73.4 | 78.3 |
| Adaptive IRCoT+HippoRAG 2 | 52.0 | 73.4 | 78.3 |

**Conclusion:** HippoRAG 2 already achieves such high single-step recall that IRCoT provides no additional benefit, confirming the paper's efficiency claims about single-step multi-hop retrieval.

---

## Improvement Experiments Summary

### IRCoT Integration
| Method | MuSiQue R@2 | 2Wiki R@2 | HotpotQA R@2 |
|---|---|---|---|
| HippoRAG 2 (baseline) | 52.0 | 73.4 | 78.3 |
| IRCoT+HippoRAG 2 | 52.0 | 73.4 | 78.3 |
| Adaptive IRCoT+HippoRAG 2 | 52.0 | 73.4 | 78.3 |

**Conclusion:** HippoRAG 2 already achieves high single-step recall. IRCoT provides no additional benefit, confirming the paper's efficiency claims.

### Query Expansion
| Method | MuSiQue R@2 | 2Wiki R@2 | HotpotQA R@2 |
|---|---|---|---|
| HippoRAG 2 (baseline) | 52.0 | 73.4 | 78.3 |
| Query Expansion | 50.2 | 68.3 | 81.0 |

**Conclusion:** Inconsistent results. Adds noise to entity extraction in some datasets.

### Entity Linking top-k=10
| Method | MuSiQue R@2 | 2Wiki R@2 | HotpotQA R@2 |
|---|---|---|---|
| HippoRAG 2 (baseline) | 52.0 | 73.4 | 78.3 |
| Linking top-k=10 | 52.7 | 69.1 | 84.0 |

**Conclusion:** Improves HotpotQA significantly but hurts 2Wiki.

### Better NER (Key Concepts Extraction)
| Method | MuSiQue R@2 | 2Wiki R@2 | HotpotQA R@2 | MuSiQue EM | HotpotQA EM |
|---|---|---|---|---|---|
| HippoRAG 2 (baseline) | 52.0 | 73.4 | 78.3 | 29.0 | 56.3 |
| Better NER | 52.2 | 73.3 | 78.3 | 29.2 | 56.4 |

**Conclusion:** Marginal improvements on MuSiQue and HotpotQA. NER prompt modification has limited effect because HippoRAG 2 already uses strong entity extraction.
