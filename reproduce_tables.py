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

    print(f"{'Method':<30} {'MuSiQue':>8} {'':>8} {'2Wiki':>8} {'':>8} {'HotpotQA':>8} {'':>8} {'Avg':>8} {'':>8}")
    print(f"{'':>30} {'R@2':>8} {'R@5':>8} {'R@2':>8} {'R@5':>8} {'R@2':>8} {'R@5':>8} {'R@2':>8} {'R@5':>8}")
    print("-"*100)

    # Paper baselines
    baselines = [
        ('BM25',                    32.3, 41.2, 51.8, 61.9, 55.4, 72.2, 46.5, 58.4),
        ('Contriever',              34.8, 46.6, 46.6, 57.5, 57.2, 75.5, 46.2, 59.9),
        ('GTR',                     37.4, 49.1, 60.2, 67.9, 59.4, 73.3, 52.3, 63.4),
        ('ColBERTv2',               37.9, 49.2, 59.2, 68.2, 64.7, 79.3, 53.9, 65.6),
        ('RAPTOR',                  35.7, 45.3, 46.3, 53.8, 58.1, 71.2, 46.7, 56.8),
        ('RAPTOR (ColBERTv2)',      36.9, 46.5, 57.3, 64.7, 63.1, 75.6, 52.4, 62.3),
        ('Proposition',             37.6, 49.3, 56.4, 63.1, 58.7, 71.1, 50.9, 61.2),
        ('Proposition (ColBERTv2)', 37.8, 50.1, 55.9, 64.9, 63.9, 78.1, 52.5, 64.4),
    ]

    for name, mr2, mr5, wr2, wr5, hr2, hr5, ar2, ar5 in baselines:
        print(f"{name:<30} {mr2:>8.1f} {mr5:>8.1f} {wr2:>8.1f} {wr5:>8.1f} {hr2:>8.1f} {hr5:>8.1f} {ar2:>8.1f} {ar5:>8.1f}")

    print("-"*100)
    # Paper HippoRAG
    print(f"{'HippoRAG (Contriever)':<30} {'41.0':>8} {'52.1':>8} {'71.5':>8} {'89.5':>8} {'59.0':>8} {'76.2':>8} {'57.2':>8} {'72.6':>8}")
    print(f"{'HippoRAG (ColBERTv2)':<30} {'40.9':>8} {'51.9':>8} {'70.7':>8} {'89.1':>8} {'60.5':>8} {'77.7':>8} {'57.4':>8} {'72.9':>8}")

    print("-"*100)
    # Our results
    m = results.get('musique', {}).get('retrieval', {})
    w = results.get('2wikimultihopqa', {}).get('retrieval', {})
    h = results.get('hotpotqa', {}).get('retrieval', {})

    mr2 = m.get('Recall@2', 0)*100
    mr5 = m.get('Recall@5', 0)*100
    wr2 = w.get('Recall@2', 0)*100
    wr5 = w.get('Recall@5', 0)*100
    hr2 = h.get('Recall@2', 0)*100
    hr5 = h.get('Recall@5', 0)*100
    ar2 = (mr2 + wr2 + hr2) / 3
    ar5 = (mr5 + wr5 + hr5) / 3

    print(f"{'HippoRAG 2 (Ours, NV-Embed-v2)':<30} {mr2:>8.1f} {mr5:>8.1f} {wr2:>8.1f} {wr5:>8.1f} {hr2:>8.1f} {hr5:>8.1f} {ar2:>8.1f} {ar5:>8.1f}")

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

    print(f"{'Method':<35} {'MuSiQue':>8} {'':>8} {'2Wiki':>8} {'':>8} {'HotpotQA':>8} {'':>8} {'Avg':>8} {'':>8}")
    print(f"{'':>35} {'R@2':>8} {'R@5':>8} {'R@2':>8} {'R@5':>8} {'R@2':>8} {'R@5':>8} {'R@2':>8} {'R@5':>8}")
    print("-"*105)

    # Paper baselines
    baselines = [
        ('IRCoT + BM25 (Default)',        34.2, 44.7, 61.2, 75.6, 65.6, 79.0, 53.7, 66.4),
        ('IRCoT + Contriever',            39.1, 52.2, 51.6, 63.8, 65.9, 81.6, 52.2, 65.9),
        ('IRCoT + ColBERTv2',             41.7, 53.7, 64.1, 74.4, 67.9, 82.0, 57.9, 70.0),
    ]

    for name, mr2, mr5, wr2, wr5, hr2, hr5, ar2, ar5 in baselines:
        print(f"{name:<35} {mr2:>8.1f} {mr5:>8.1f} {wr2:>8.1f} {wr5:>8.1f} {hr2:>8.1f} {hr5:>8.1f} {ar2:>8.1f} {ar5:>8.1f}")

    print("-"*105)
    # Paper HippoRAG
    print(f"{'IRCoT+HippoRAG (Contriever)':<35} {'43.9':>8} {'56.6':>8} {'75.3':>8} {'93.4':>8} {'65.8':>8} {'82.3':>8} {'61.7':>8} {'77.4':>8}")
    print(f"{'IRCoT+HippoRAG (ColBERTv2)':<35} {'45.3':>8} {'57.6':>8} {'75.8':>8} {'93.9':>8} {'67.0':>8} {'83.0':>8} {'62.7':>8} {'78.2':>8}")

    print("-"*105)
    # Our results
    m = results.get('musique', {}).get('retrieval', {})
    w = results.get('2wikimultihopqa', {}).get('retrieval', {})
    h = results.get('hotpotqa', {}).get('retrieval', {})

    if m and w and h:
        mr2 = m.get('Recall@2', 0)*100
        mr5 = m.get('Recall@5', 0)*100
        wr2 = w.get('Recall@2', 0)*100
        wr5 = w.get('Recall@5', 0)*100
        hr2 = h.get('Recall@2', 0)*100
        hr5 = h.get('Recall@5', 0)*100
        ar2 = (mr2 + wr2 + hr2) / 3
        ar5 = (mr5 + wr5 + hr5) / 3
        print(f"{'IRCoT+HippoRAG 2 (Ours)':<35} {mr2:>8.1f} {mr5:>8.1f} {wr2:>8.1f} {wr5:>8.1f} {hr2:>8.1f} {hr5:>8.1f} {ar2:>8.1f} {ar5:>8.1f}")
    else:
        print(f"{'IRCoT+HippoRAG 2 (Ours)':<35} {'TBD':>8} {'TBD':>8} {'TBD':>8} {'TBD':>8} {'TBD':>8} {'TBD':>8} {'TBD':>8} {'TBD':>8}")

def reproduce_table_4():
    """Reproduce Table 4: QA Performance"""
    print("\n" + "="*60)
    print("TABLE 4: QA Performance")
    print("="*60)

    datasets = ['musique', '2wikimultihopqa', 'hotpotqa']
    results = {}
    ircot_results = {}
    for ds in datasets:
        path = f'outputs/{ds}/results_meta-llama_Llama-3.1-8B-Instruct.json'
        if os.path.exists(path):
            with open(path) as f:
                results[ds] = json.load(f)
        ircot_path = f'outputs/{ds}/results_ircot_meta-llama_Llama-3.1-8B-Instruct.json'
        if os.path.exists(ircot_path):
            with open(ircot_path) as f:
                ircot_results[ds] = json.load(f)

    print(f"{'Method':<35} {'MuSiQue':>8} {'':>8} {'2Wiki':>8} {'':>8} {'HotpotQA':>8} {'':>8} {'Avg':>8} {'':>8}")
    print(f"{'':>35} {'EM':>8} {'F1':>8} {'EM':>8} {'F1':>8} {'EM':>8} {'F1':>8} {'EM':>8} {'F1':>8}")
    print("-"*105)

    # Paper baselines
    baselines = [
        ('None (Paper)',                  12.5, 24.1, 31.0, 39.6, 30.4, 42.8, 24.6, 35.5),
        ('ColBERTv2 (Paper)',             15.5, 26.4, 33.4, 43.3, 43.4, 57.7, 30.8, 42.5),
        ('HippoRAG (ColBERTv2, Paper)',   19.2, 29.8, 46.6, 59.5, 41.8, 55.0, 35.9, 48.1),
        ('IRCoT (ColBERTv2, Paper)',      19.1, 30.5, 35.4, 45.1, 45.5, 58.4, 33.3, 44.7),
        ('IRCoT+HippoRAG (ColBERTv2)',    21.9, 33.3, 47.7, 62.7, 45.7, 59.2, 38.4, 51.7),
    ]

    for name, mem, mf1, wem, wf1, hem, hf1, aem, af1 in baselines:
        print(f"{name:<35} {mem:>8.1f} {mf1:>8.1f} {wem:>8.1f} {wf1:>8.1f} {hem:>8.1f} {hf1:>8.1f} {aem:>8.1f} {af1:>8.1f}")

    print("-"*105)
    # Our results
    m = results.get('musique', {}).get('qa', {})
    w = results.get('2wikimultihopqa', {}).get('qa', {})
    h = results.get('hotpotqa', {}).get('qa', {})

    mem = m.get('ExactMatch', 0)*100
    mf1 = m.get('F1', 0)*100
    wem = w.get('ExactMatch', 0)*100
    wf1 = w.get('F1', 0)*100
    hem = h.get('ExactMatch', 0)*100
    hf1 = h.get('F1', 0)*100
    aem = (mem + wem + hem) / 3
    af1 = (mf1 + wf1 + hf1) / 3
    print(f"{'HippoRAG 2 (Ours)':<35} {mem:>8.1f} {mf1:>8.1f} {wem:>8.1f} {wf1:>8.1f} {hem:>8.1f} {hf1:>8.1f} {aem:>8.1f} {af1:>8.1f}")

    # IRCoT results
    im = ircot_results.get('musique', {}).get('qa', {})
    iw = ircot_results.get('2wikimultihopqa', {}).get('qa', {})
    ih = ircot_results.get('hotpotqa', {}).get('qa', {})
    if im and iw and ih:
        imem = im.get('ExactMatch', 0)*100
        imf1 = im.get('F1', 0)*100
        iwem = iw.get('ExactMatch', 0)*100
        iwf1 = iw.get('F1', 0)*100
        ihem = ih.get('ExactMatch', 0)*100
        ihf1 = ih.get('F1', 0)*100
        iaem = (imem + iwem + ihem) / 3
        iaf1 = (imf1 + iwf1 + ihf1) / 3
        print(f"{'IRCoT+HippoRAG 2 (Ours)':<35} {imem:>8.1f} {imf1:>8.1f} {iwem:>8.1f} {iwf1:>8.1f} {ihem:>8.1f} {ihf1:>8.1f} {iaem:>8.1f} {iaf1:>8.1f}")
    else:
        print(f"{'IRCoT+HippoRAG 2 (Ours)':<35} {'TBD':>8} {'TBD':>8} {'TBD':>8} {'TBD':>8} {'TBD':>8} {'TBD':>8} {'TBD':>8} {'TBD':>8}")

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

def reproduce_table_6():
    """Reproduce Table 6: All-Recall Metric"""
    print("\n" + "="*60)
    print("TABLE 6: All-Recall Metric (AR@2, AR@5)")
    print("="*60)

    datasets = ['musique', '2wikimultihopqa', 'hotpotqa']
    results = {}
    for ds in datasets:
        path = f'outputs/{ds}/results_meta-llama_Llama-3.1-8B-Instruct.json'
        if os.path.exists(path):
            with open(path) as f:
                results[ds] = json.load(f)

    print(f"{'Method':<25} {'MuSiQue':>8} {'':>8} {'2Wiki':>8} {'':>8} {'HotpotQA':>8} {'':>8} {'Avg':>8} {'':>8}")
    print(f"{'':>25} {'AR@2':>8} {'AR@5':>8} {'AR@2':>8} {'AR@5':>8} {'AR@2':>8} {'AR@5':>8} {'AR@2':>8} {'AR@5':>8}")
    print("-"*95)

    # Paper values
    print(f"{'ColBERTv2 (Paper)':<25} {6.8:>8.1f} {16.1:>8.1f} {25.1:>8.1f} {37.1:>8.1f} {33.3:>8.1f} {59.0:>8.1f} {21.7:>8.1f} {37.4:>8.1f}")
    print(f"{'HippoRAG (Paper)':<25} {10.2:>8.1f} {22.4:>8.1f} {45.4:>8.1f} {75.7:>8.1f} {33.8:>8.1f} {57.9:>8.1f} {29.8:>8.1f} {52.0:>8.1f}")
    print("-"*95)

    # Our results
    m = results.get('musique', {}).get('all_recall', {})
    w = results.get('2wikimultihopqa', {}).get('all_recall', {})
    h = results.get('hotpotqa', {}).get('all_recall', {})

    if m and w and h:
        mar2 = (m.get('AR@2') or 0) * 100
        mar5 = (m.get('AR@5') or 0) * 100
        war2 = (w.get('AR@2') or 0) * 100
        war5 = (w.get('AR@5') or 0) * 100
        har2 = (h.get('AR@2') or 0) * 100
        har5 = (h.get('AR@5') or 0) * 100
        aar2 = (mar2 + war2 + har2) / 3
        aar5 = (mar5 + war5 + har5) / 3
        print(f"{'HippoRAG 2 (Ours)':<25} {mar2:>8.1f} {mar5:>8.1f} {war2:>8.1f} {war5:>8.1f} {har2:>8.1f} {har5:>8.1f} {aar2:>8.1f} {aar5:>8.1f}")
    else:
        print(f"{'HippoRAG 2 (Ours)':<25} {'TBD':>8} {'TBD':>8} {'TBD':>8} {'TBD':>8} {'TBD':>8} {'TBD':>8} {'TBD':>8} {'TBD':>8}")
        print("\nNote: Running new jobs with all_recall computation.")

def reproduce_table_7():
    """Reproduce Table 7: Multi-hop question types"""
    print("\n" + "="*60)
    print("TABLE 7: Multi-hop Question Types")
    print("="*60)

    import os
    os.environ['OPENAI_API_KEY'] = 'local'
    os.environ['HF_HOME'] = '/scratch/coa258/hf_cache'

    from src.hipporag.HippoRAG import HippoRAG
    from src.hipporag.utils.config_utils import BaseConfig

    config = BaseConfig(
        llm_name='meta-llama/Llama-3.1-8B-Instruct',
        llm_base_url='http://192.168.1.210:8000/v1',
        embedding_model_name='nvidia/NV-Embed-v2',
        dataset='musique',
    )

    hipporag = HippoRAG(global_config=config)

    queries = [
        'In which district was Alhandra born?',
        "Which Stanford professor works on the neuroscience of Alzheimer's?"
    ]

    paper_hipporag = [
        ['Alhandra', 'Vila de Xira', 'Portugal'],
        ['Thomas Südhof', 'Karl Deisseroth', 'Robert Sapolsky'],
    ]

    paper_colbert = [
        ['Alhandra', 'Dimuthu Abayakoon', "Ja'ar"],
        ['Brian Knutson', 'Eric Knudsen', 'Lisa Giocomo'],
    ]

    paper_ircot = [
        ['Alhandra', 'Vila de Xira', 'Póvoa de Santa Iria'],
        ['Brian Knutson', 'Eric Knudsen', 'Lisa Giocomo'],
    ]

    results = hipporag.retrieve(queries=queries, num_to_retrieve=3)

    for i, (query, result) in enumerate(zip(queries, results)):
        print(f"\nQ: {query}")
        print(f"{'Method':<20} {'Top-1':<30} {'Top-2':<30} {'Top-3':<30}")
        print("-"*115)
        our_docs = [doc.split('\n')[0][:28] for doc in result.docs[:3]]
        while len(our_docs) < 3:
            our_docs.append('N/A')
        print(f"{'HippoRAG 2 (Ours)':<20} {our_docs[0]:<30} {our_docs[1]:<30} {our_docs[2]:<30}")
        print(f"{'HippoRAG (Paper)':<20} {paper_hipporag[i][0]:<30} {paper_hipporag[i][1]:<30} {paper_hipporag[i][2]:<30}")
        print(f"{'ColBERTv2 (Paper)':<20} {paper_colbert[i][0]:<30} {paper_colbert[i][1]:<30} {paper_colbert[i][2]:<30}")
        print(f"{'IRCoT (Paper)':<20} {paper_ircot[i][0]:<30} {paper_ircot[i][1]:<30} {paper_ircot[i][2]:<30}")


if __name__ == "__main__":
    reproduce_table_1()
    reproduce_table_2()
    reproduce_table_3()
    reproduce_table_4()
    reproduce_table_5()
    reproduce_table_6()
    reproduce_table_7()
    print("\nDone!")
