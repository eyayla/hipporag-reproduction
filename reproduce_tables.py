import json
import pickle
import os

def reproduce_table_1():
    """Reproduce Table 1: Dataset and KG statistics"""
    print("\n" + "="*60)
    print("TABLE 1: Dataset and Knowledge Graph Statistics")
    print("="*60)

    datasets = [
        ('MuSiQue', 'outputs/musique/meta-llama_Llama-3.1-8B-Instruct_nvidia_NV-Embed-v2'),
        ('HotpotQA', 'outputs/hotpotqa/meta-llama_Llama-3.1-8B-Instruct_nvidia_NV-Embed-v2'),
        ('2Wiki', 'outputs/2wikimultihopqa/meta-llama_Llama-3.1-8B-Instruct_nvidia_NV-Embed-v2'),
    ]

    print(f"{'Metric':<25} {'MuSiQue':>12} {'2Wiki':>12} {'HotpotQA':>12}")
    print("-"*65)

    passages, nodes, edges = {}, {}, {}
    for name, path in datasets:
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

    print(f"{'# of Passages':<25} {passages['MuSiQue']:>12,} {passages['2Wiki']:>12,} {passages['HotpotQA']:>12,}")
    print(f"{'# of Unique Nodes':<25} {nodes['MuSiQue']:>12,} {nodes['2Wiki']:>12,} {nodes['HotpotQA']:>12,}")
    print(f"{'# of Unique Edges':<25} {edges['MuSiQue']:>12,} {edges['2Wiki']:>12,} {edges['HotpotQA']:>12,}")

    print("\nPaper values (HippoRAG 1):")
    print(f"{'# of Passages':<25} {'11,656':>12} {'6,119':>12} {'9,221':>12}")
    print(f"{'# of Unique Nodes':<25} {'91,729':>12} {'42,694':>12} {'82,157':>12}")
    print(f"{'# of Unique Edges':<25} {'21,714':>12} {'7,867':>12} {'17,523':>12}")

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

if __name__ == "__main__":
    reproduce_table_1()
    reproduce_table_2()
    reproduce_table_3()
    print("\nDone!")
