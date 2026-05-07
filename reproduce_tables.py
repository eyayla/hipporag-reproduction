import json
import pickle
import os

def reproduce_table_1():
    """Reproduce Table 1: Dataset and KG statistics"""
    print("\n" + "="*60)
    print("TABLE 1: Dataset and Knowledge Graph Statistics")
    print("="*60)

    datasets = [
        ('MuSiQue', 'outputs/musique/meta-llama_Llama-3.1-8B-Instruct_nvidia_NV-Embed-v2', 'musique'),
        ('2Wiki', 'outputs/2wikimultihopqa/meta-llama_Llama-3.1-8B-Instruct_nvidia_NV-Embed-v2', '2wikimultihopqa'),
        ('HotpotQA', 'outputs/hotpotqa/meta-llama_Llama-3.1-8B-Instruct_nvidia_NV-Embed-v2', 'hotpotqa'),
    ]

    passages, nodes, edges, triples, synonym_edges = {}, {}, {}, {}, {}

    for name, path, ds in datasets:
        with open(f'{path}/graph.pickle', 'rb') as f:
            G = pickle.load(f)

        passage_ids = set(v.index for v in G.vs if v['hash_id'].startswith('chunk-'))
        entity_ids  = set(v.index for v in G.vs if v['hash_id'].startswith('entity-'))
        ee_edge_pairs = set()
        for e in G.es:
            s, t = e.source, e.target
            if s in entity_ids and t in entity_ids:
                ee_edge_pairs.add((min(s,t), max(s,t)))

        passages[name] = len(passage_ids)
        nodes[name] = len(entity_ids)
        edges[name] = len(ee_edge_pairs)

        with open(f'outputs/{ds}/openie_results_ner_meta-llama_Llama-3.1-8B-Instruct.json') as f:
            openie_data = json.load(f)

        t_set = set()
        triple_count = 0
        for chunk in openie_data['docs']:
            if isinstance(chunk, dict):
                for t in chunk.get('extracted_triples', []):
                    if isinstance(t, list) and len(t) == 3:
                        t_set.add(tuple(t))
                        triple_count += 1
        triples[name] = len(t_set)
        synonym_edges[name] = G.ecount() - triple_count

    print(f"{'Metric':<30} {'MuSiQue':>12} {'2Wiki':>12} {'HotpotQA':>12}")
    print("-"*70)
    print(f"{'# of Passages':<30} {passages['MuSiQue']:>12,} {passages['2Wiki']:>12,} {passages['HotpotQA']:>12,}")
    print(f"{'# of Unique Nodes':<30} {nodes['MuSiQue']:>12,} {nodes['2Wiki']:>12,} {nodes['HotpotQA']:>12,}")
    print(f"{'# of Unique Edges':<30} {edges['MuSiQue']:>12,} {edges['2Wiki']:>12,} {edges['HotpotQA']:>12,}")
    print(f"{'# of Unique Triples':<30} {triples['MuSiQue']:>12,} {triples['2Wiki']:>12,} {triples['HotpotQA']:>12,}")
    print(f"{'NV-Embed-v2 Synonym Edges':<30} {synonym_edges['MuSiQue']:>12,} {synonym_edges['2Wiki']:>12,} {synonym_edges['HotpotQA']:>12,}")

    print("\n-- Paper values (HippoRAG 1, GPT-3.5 + ColBERTv2) --")
    print(f"{'# of Passages':<30} {'11,656':>12} {'6,119':>12} {'9,221':>12}")
    print(f"{'# of Unique Nodes':<30} {'91,729':>12} {'42,694':>12} {'82,157':>12}")
    print(f"{'# of Unique Edges':<30} {'21,714':>12} {'7,867':>12} {'17,523':>12}")
    print(f"{'# of Unique Triples':<30} {'107,448':>12} {'50,671':>12} {'98,709':>12}")
    print(f"{'ColBERTv2 Synonym Edges':<30} {'191,636':>12} {'82,526':>12} {'171,856':>12}")

def reproduce_table_2():
    """Reproduce Table 2: Single-step retrieval performance"""
    print("\n" + "="*60)
    print("TABLE 2: Single-step Retrieval Performance")
    print("="*60)

    datasets = ['musique', '2wikimultihopqa', 'hotpotqa']
    results = {}
    for ds in datasets:
        path = f'outputs/{ds}/results_meta-llama_Llama-3.1-8B-Instruct.json'
        if os.path.exists(path):
            with open(path) as f:
                results[ds] = json.load(f)

    print(f"{'Method':<30} {'MuSiQue R@2':>12} {'MuSiQue R@5':>12} {'2Wiki R@2':>10} {'2Wiki R@5':>10} {'HotpotQA R@2':>13} {'HotpotQA R@5':>13}")
    print("-"*100)

    # Paper values
    print(f"{'HippoRAG (Paper, Contriever)':<30} {'41.0':>12} {'52.1':>12} {'71.5':>10} {'89.5':>10} {'59.0':>13} {'76.2':>13}")

    # Our values
    m  = results.get('musique', {}).get('retrieval', {})
    w  = results.get('2wikimultihopqa', {}).get('retrieval', {})
    h  = results.get('hotpotqa', {}).get('retrieval', {})
    print(f"{'HippoRAG 2 (Ours)':<30} {m.get('Recall@2',0)*100:>12.1f} {m.get('Recall@5',0)*100:>12.1f} {w.get('Recall@2',0)*100:>10.1f} {w.get('Recall@5',0)*100:>10.1f} {h.get('Recall@2',0)*100:>13.1f} {h.get('Recall@5',0)*100:>13.1f}")

def reproduce_table_3():
    """Reproduce Table 3: Multi-step retrieval performance (IRCoT)"""
    print("\n" + "="*60)
    print("TABLE 3: Multi-step Retrieval Performance (IRCoT)")
    print("="*60)

    datasets = ['musique', '2wikimultihopqa', 'hotpotqa']
    results = {}
    for ds in datasets:
        path = f'outputs/{ds}/results_ircot_meta-llama_Llama-3.1-8B-Instruct.json'
        if os.path.exists(path):
            with open(path) as f:
                results[ds] = json.load(f)

    print(f"{'Method':<35} {'MuSiQue R@2':>12} {'MuSiQue R@5':>12} {'2Wiki R@2':>10} {'2Wiki R@5':>10} {'HotpotQA R@2':>13} {'HotpotQA R@5':>13}")
    print("-"*105)

    # Paper values
    print(f"{'IRCoT+HippoRAG (Paper, Contriever)':<35} {'43.9':>12} {'56.6':>12} {'75.3':>10} {'93.4':>10} {'65.8':>13} {'82.3':>13}")

    # Our values
    m = results.get('musique', {}).get('retrieval', {})
    w = results.get('2wikimultihopqa', {}).get('retrieval', {})
    h = results.get('hotpotqa', {}).get('retrieval', {})

    if m:
        print(f"{'IRCoT+HippoRAG 2 (Ours)':<35} {m.get('Recall@2',0)*100:>12.1f} {m.get('Recall@5',0)*100:>12.1f} {w.get('Recall@2',0)*100:>10.1f} {w.get('Recall@5',0)*100:>10.1f} {h.get('Recall@2',0)*100:>13.1f} {h.get('Recall@5',0)*100:>13.1f}")
    else:
        print(f"{'IRCoT+HippoRAG 2 (Ours)':<35} {'TBD':>12} {'TBD':>12} {'TBD':>10} {'TBD':>10} {'TBD':>13} {'TBD':>13}")

def reproduce_table_4():
    """Reproduce Table 4: QA Performance"""
    print("\n" + "="*60)
    print("TABLE 4: QA Performance")
    print("="*60)

    datasets = ['musique', '2wikimultihopqa', 'hotpotqa']
    results = {}
    for ds in datasets:
        path = f'outputs/{ds}/results_meta-llama_Llama-3.1-8B-Instruct.json'
        if os.path.exists(path):
            with open(path) as f:
                results[ds] = json.load(f)

    print(f"{'Method':<30} {'MuSiQue EM':>10} {'MuSiQue F1':>10} {'2Wiki EM':>8} {'2Wiki F1':>8} {'HotpotQA EM':>11} {'HotpotQA F1':>11}")
    print("-"*95)

    # Paper values
    print(f"{'HippoRAG (Paper, ColBERTv2)':<30} {'19.2':>10} {'29.8':>10} {'46.6':>8} {'59.5':>8} {'41.8':>11} {'55.0':>11}")

    # Our values
    m = results.get('musique', {}).get('qa', {})
    w = results.get('2wikimultihopqa', {}).get('qa', {})
    h = results.get('hotpotqa', {}).get('qa', {})
    print(f"{'HippoRAG 2 (Ours)':<30} {m.get('ExactMatch',0)*100:>10.1f} {m.get('F1',0)*100:>10.1f} {w.get('ExactMatch',0)*100:>8.1f} {w.get('F1',0)*100:>8.1f} {h.get('ExactMatch',0)*100:>11.1f} {h.get('F1',0)*100:>11.1f}")

def reproduce_table_5():
    """Reproduce Table 5: Ablation Study"""
    print("\n" + "="*60)
    print("TABLE 5: Dissecting HippoRAG (Ablation Study)")
    print("="*60)

    print(f"{'Method':<35} {'MuSiQue R@2':>12} {'MuSiQue R@5':>12} {'2Wiki R@2':>10} {'2Wiki R@5':>10} {'HotpotQA R@2':>13} {'HotpotQA R@5':>13}")
    print("-"*110)

    # Paper values
    print("-- Paper values (HippoRAG 1, ColBERTv2) --")
    print(f"{'HippoRAG (baseline)':<35} {'40.9':>12} {'51.9':>12} {'70.7':>10} {'89.1':>10} {'60.5':>13} {'77.7':>13}")
    print(f"{'OpenIE: Llama-3.1-8B':<35} {'40.8':>12} {'51.9':>12} {'62.5':>10} {'77.5':>10} {'59.9':>13} {'75.1':>13}")
    print(f"{'OpenIE: Llama-3.1-70B':<35} {'41.8':>12} {'53.7':>12} {'68.8':>10} {'85.3':>10} {'60.8':>13} {'78.6':>13}")

    print("\n-- Our values (HippoRAG 2, Llama-3.1-8B OpenIE) --")
    # Our results use Llama-3.1-8B for OpenIE
    datasets = ['musique', '2wikimultihopqa', 'hotpotqa']
    results = {}
    for ds in datasets:
        path = f'outputs/{ds}/results_meta-llama_Llama-3.1-8B-Instruct.json'
        if os.path.exists(path):
            with open(path) as f:
                results[ds] = json.load(f)

    m = results.get('musique', {}).get('retrieval', {})
    w = results.get('2wikimultihopqa', {}).get('retrieval', {})
    h = results.get('hotpotqa', {}).get('retrieval', {})
    print(f"{'OpenIE: Llama-3.1-8B (Ours)':<35} {m.get('Recall@2',0)*100:>12.1f} {m.get('Recall@5',0)*100:>12.1f} {w.get('Recall@2',0)*100:>10.1f} {w.get('Recall@5',0)*100:>10.1f} {h.get('Recall@2',0)*100:>13.1f} {h.get('Recall@5',0)*100:>13.1f}")
    print("\nNote: Full ablation study (PPR alternatives, w/o Node Specificity)")
    print("requires additional experiments not yet completed.")


if __name__ == "__main__":
    reproduce_table_1()
    reproduce_table_2()
    reproduce_table_3()
    reproduce_table_4()
    reproduce_table_5()
    print("\nDone!")
