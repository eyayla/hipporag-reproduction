import os
import json
import logging

import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import argparse
from typing import List
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import string_to_bool
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.qa_utils import reason_step
from src.hipporag.prompts.prompt_template_manager import PromptTemplateManager
from src.hipporag.llm.openai_gpt import CacheOpenAI

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_gold_docs(samples, dataset_name=None):
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            gold_paragraphs = [item for item in sample['paragraphs'] if item.get('is_supporting', True)]
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]
        gold_docs.append(list(set(gold_doc)))
    return gold_docs

def get_gold_answers(samples):
    gold_answers = []
    for sample in samples:
        if 'answer' in sample:
            gold_answers.append(sample['answer'])
        elif 'gold_ans' in sample:
            gold_answers.append(sample['gold_ans'])
        else:
            gold_answers.append('')
    return gold_answers

def ircot_retrieve(hipporag, query, passages, thoughts, max_steps=2):
    """IRCoT: iterative retrieval with chain-of-thought reasoning"""
    prompt_manager = hipporag.prompt_template_manager
    llm_client = hipporag.llm_model

    current_query = query
    all_passages = list(passages)

    for step in range(max_steps):
        # LLM ile bir sonraki düşünceyi üret
        ircot_dataset = hipporag.global_config.dataset if hipporag.global_config.dataset in ['musique', 'hotpotqa'] else 'musique'
        thought = reason_step(
            dataset=ircot_dataset,
            prompt_template_manager=prompt_manager,
            query=query,
            passages=all_passages,
            thoughts=thoughts,
            llm_client=llm_client
        )

        if not thought:
            break

        thoughts.append(thought)

        # Eğer son adıma geldiyse dur
        if "so the answer is" in thought.lower():
            break

        # Yeni düşünceyi sorgu olarak kullan ve tekrar retrieve et
        new_results = hipporag.retrieve(queries=[thought], num_to_retrieve=5)
        if new_results:
            for doc in new_results[0].docs[:5]:
                if doc not in all_passages:
                    all_passages.append(doc)

    return all_passages, thoughts

def main():
    parser = argparse.ArgumentParser(description='HippoRAG IRCoT retrieval and QA')
    parser.add_argument('--dataset', type=str, default='sample')
    parser.add_argument('--llm_base_url', type=str, default=None)
    parser.add_argument('--llm_name', type=str, default='gpt-4o-mini')
    parser.add_argument('--embedding_name', type=str, default='nvidia/NV-Embed-v2')
    parser.add_argument('--max_steps', type=int, default=2)
    args = parser.parse_args()

    # Dataset yükle
    dataset_path = f'reproduce/dataset/{args.dataset}.json'
    corpus_path = f'reproduce/dataset/{args.dataset}_corpus.json'

    with open(dataset_path) as f:
        samples = json.load(f)
    with open(corpus_path) as f:
        corpus = json.load(f)

    docs = [item['title'] + '\n' + item['text'] for item in corpus]
    all_queries = [s['question'] for s in samples]
    gold_docs = get_gold_docs(samples, args.dataset)
    gold_answers = get_gold_answers(samples)

    # HippoRAG başlat
    config = BaseConfig(
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_name,
        dataset=args.dataset,
    )

    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs)

    # Retrieval evaluation
    from src.hipporag.evaluation.retrieval_eval import RetrievalRecallEvaluator
    evaluator = RetrievalRecallEvaluator()
    k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
    ircot_retrieval_result, _ = evaluator.calculate_metric_scores(
        gold_docs=gold_docs,
        retrieved_docs=all_retrieved,
        k_list=k_list
    )
    ircot_retrieval_result = {k: round(float(v), 4) for k, v in ircot_retrieval_result.items()}
    logger.info(f"IRCoT Retrieval results: {ircot_retrieval_result}")
 
    # IRCoT retrieval + QA
    logger.info("Starting IRCoT retrieval...")
    all_retrieved = []

    for i, query in enumerate(all_queries):
        # İlk retrieval
        initial_results = hipporag.retrieve(queries=[query], num_to_retrieve=5)
        initial_passages = []
        if initial_results:
            initial_passages = initial_results[0].docs[:5]

        # IRCoT ile genişlet
        final_passages, thoughts = ircot_retrieve(
            hipporag=hipporag,
            query=query,
            passages=initial_passages,
            thoughts=[],
            max_steps=args.max_steps
        )
        all_retrieved.append(final_passages)

        if i % 100 == 0:
            logger.info(f"Processed {i}/{len(all_queries)} queries")

    # QA - IRCoT retrieved passages kullanarak
    logger.info("Starting QA...")
    from src.hipporag.utils.misc_utils import QuerySolution
    
    # IRCoT sonuçlarını QuerySolution'a dönüştür
    ircot_solutions = []
    for i, query in enumerate(all_queries):
        qs = QuerySolution(question=query, docs=all_retrieved[i], doc_scores=np.array([1.0]*len(all_retrieved[i])))
        ircot_solutions.append(qs)
 
    result = hipporag.rag_qa(queries=ircot_solutions, gold_docs=gold_docs, gold_answers=gold_answers)
    if result is not None:
        if len(result) >= 5:
            solutions, _, _, retrieval_result, qa_results = result
        else:
            solutions, _, _ = result
            retrieval_result = None
            # QA sonuçlarını manuel hesapla
            from src.hipporag.evaluation.qa_eval import QAEvaluator
            evaluator = QAEvaluator()
            qa_results = evaluator.evaluate(solutions, gold_answers)
        
        # Debug
        for s in solutions[:2]:
            print(f"Q: {s.question}")
            print(f"A: {s.answer}")
            print(f"Gold: {s.gold_answers}")
        
        out = {
            'dataset': args.dataset,
            'llm': args.llm_name,
            'embedding': args.embedding_name,
            'method': 'IRCoT+HippoRAG',
            'max_steps': args.max_steps,
            'retrieval':ircot_retrieval_result,
            'qa': qa_results
        }
        out_path = f'outputs/{args.dataset}/results_ircot_{args.llm_name.replace("/","_")}.json'
        os.makedirs(f'outputs/{args.dataset}', exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'Results saved to {out_path}')

if __name__ == "__main__":
    main()
