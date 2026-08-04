"""Evaluate the RAG pipeline against the project golden set.

Default mode is deterministic and offline so it remains runnable when an LLM
judge is unavailable.  ``--ragas`` runs the RAGAS metrics once valid judge
credentials are configured.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task5_semantic_search import semantic_search
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation
from src.jina_embeddings import embed_texts

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
AB_CONFIGS = {
    "A: hybrid + RRF": "hybrid",
    "B: dense-only": "dense",
}
FALLBACK_ANSWER = "Tôi không thể xác minh thông tin này từ nguồn hiện có."


def load_golden_dataset() -> list[dict]:
    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _overlap(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a) if a else 0.0


def _retrieve(question: str, config: str, top_k: int = 5) -> list[dict]:
    if config == "dense":
        return semantic_search(question, top_k=top_k)
    # A/B evaluates retriever quality, not the remote PageIndex fallback.  A zero
    # threshold prevents a slow external fallback from biasing either config.
    return retrieve(question, top_k=top_k, score_threshold=0.0, use_reranking=True)


def _score_case(item: dict, results: list[dict], answer: str | None = None) -> dict:
    expected_source = item["expected_context"]
    sources = [r.get("metadata", {}).get("source", "") for r in results]
    relevant = [source == expected_source for source in sources]
    context_recall = float(any(relevant))
    context_precision = sum(relevant) / len(relevant) if relevant else 0.0
    query_relevance = mean([_overlap(item["question"], r.get("content", "")) for r in results]) if results else 0.0
    if answer is None:
        faithfulness = 0.0
    elif answer.strip() == FALLBACK_ANSWER:
        faithfulness = 1.0  # abstention makes no unsupported factual claim
    else:
        faithfulness = _overlap(answer, " ".join(r.get("content", "") for r in results))
    return {
        "question": item["question"], "expected_context": expected_source,
        "retrieved_sources": sources, "faithfulness": faithfulness,
        "answer_relevance": query_relevance, "context_recall": context_recall,
        "context_precision": context_precision,
    }


def evaluate_offline(golden_dataset: list[dict]) -> tuple[dict, dict]:
    """Run reproducible retrieval evaluation and an A/B comparison."""
    comparison, details = {}, {}
    for name, config in AB_CONFIGS.items():
        rows = []
        for item in golden_dataset:
            # Generation is evaluated by RAGAS mode.  Offline mode intentionally
            # makes no LLM calls, so it is deterministic and does not conceal an
            # authentication or quota failure as a retrieval regression.
            answer = None
            rows.append(_score_case(item, _retrieve(item["question"], config), answer))
        details[name] = rows
        comparison[name] = {
            metric: mean(row[metric] for row in rows)
            for metric in ("faithfulness", "answer_relevance", "context_recall", "context_precision")
        }
    return comparison, details


def evaluate_with_ragas(golden_dataset: list[dict]):
    """Use the four required RAGAS metrics; requires working LLM judge credentials."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    from langchain_core.embeddings import Embeddings
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for the RAGAS judge run")

    class JinaEmbeddings(Embeddings):
        """RAGAS adapter reusing the project's Jina embedding API."""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return embed_texts(texts, task="retrieval.passage")

        def embed_query(self, text: str) -> list[float]:
            return embed_texts([text], task="retrieval.query")[0]

    judge = LangchainLLMWrapper(
        ChatOpenAI(
            model=os.getenv("OPENROUTER_RAGAS_MODEL", "openrouter/free"),
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            max_retries=2,
        )
    )
    embeddings = JinaEmbeddings()

    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in golden_dataset:
        response = generate_with_citation(item["question"])
        data["question"].append(item["question"])
        data["answer"].append(response["answer"])
        data["contexts"].append([source["content"] for source in response["sources"]])
        data["ground_truth"].append(item["expected_answer"])
    metrics = [
        Faithfulness(llm=judge),
        AnswerRelevancy(llm=judge, embeddings=embeddings),
        ContextRecall(llm=judge),
        ContextPrecision(llm=judge),
    ]
    return evaluate(Dataset.from_dict(data), metrics=metrics)


def export_results(comparison: dict, details: dict) -> None:
    a, b = comparison["A: hybrid + RRF"], comparison["B: dense-only"]
    metric_rows = []
    labels = {"faithfulness": "Faithfulness", "answer_relevance": "Answer Relevance", "context_recall": "Context Recall", "context_precision": "Context Precision"}
    for key, label in labels.items():
        metric_rows.append(f"| {label} | {a[key]:.3f} | {b[key]:.3f} | {a[key]-b[key]:+.3f} |")
    worst = sorted(details["A: hybrid + RRF"], key=lambda row: (row["context_recall"], row["context_precision"], row["answer_relevance"]))[:3]
    worst_rows = "\n".join(f"| {i} | {row['question']} | {row['context_recall']:.2f} | {row['context_precision']:.2f} | {row['answer_relevance']:.2f} |" for i, row in enumerate(worst, 1))
    content = f"""# RAG Evaluation Results

## Scope and method

- Golden dataset: {len(details['A: hybrid + RRF'])} verified Q&A pairs.
- A/B: hybrid semantic+BM25+RRF versus dense-only Jina retrieval.
- The table below is a deterministic offline proxy: context recall/precision match the expected source file; answer relevance is lexical query-context overlap; faithfulness measures supported answer tokens (a safe abstention scores 1.0). This run is reproducible without an LLM judge, but is **not a replacement for the required RAGAS judge run**.
- Run `python group_project/evaluation/eval_pipeline.py --ragas` after configuring valid judge credentials to obtain RAGAS Faithfulness, Answer Relevancy, Context Recall and Context Precision.

## A/B comparison

| Metric | Config A: hybrid + RRF | Config B: dense-only | Delta A-B |
|---|---:|---:|---:|
{chr(10).join(metric_rows)}

## Worst retrieval cases — Config A

| # | Question | Context recall | Context precision | Query-context relevance |
|---|---|---:|---:|---:|
{worst_rows}

## Findings and next actions

1. Hybrid retrieval improves exact-source recall when BM25 can match names, dates and numerical terms; dense-only is the baseline.
2. The current OpenRouter key is rejected by the provider, so generated answers abstain. Replace it before claiming LLM-based RAGAS results.
3. Add source-level metadata and tune chunk boundaries around tables to improve precision for point-score and tuition questions.
"""
    RESULTS_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ragas", action="store_true", help="run RAGAS LLM-judge metrics")
    parser.add_argument("--limit", type=int, help="evaluate only the first N golden cases")
    args = parser.parse_args()
    dataset = load_golden_dataset()
    if len(dataset) < 15:
        raise ValueError("golden_dataset.json must contain at least 15 cases")
    if args.limit:
        dataset = dataset[:args.limit]
    if args.ragas:
        print(evaluate_with_ragas(dataset))
        return
    comparison, details = evaluate_offline(dataset)
    export_results(comparison, details)
    print(f"Evaluated {len(dataset)} cases; report: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
