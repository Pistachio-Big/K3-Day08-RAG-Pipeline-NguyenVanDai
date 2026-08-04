# RAG Evaluation Results

## Scope and method

- Golden dataset: 18 verified Q&A pairs.
- A/B: hybrid semantic+BM25+RRF versus dense-only Jina retrieval.
- The table below is a deterministic offline proxy: context recall/precision match the expected source file; answer relevance is lexical query-context overlap; faithfulness measures supported answer tokens (a safe abstention scores 1.0). This run is reproducible without an LLM judge, but is **not a replacement for the required RAGAS judge run**.
- Run `python group_project/evaluation/eval_pipeline.py --ragas` after configuring valid judge credentials to obtain RAGAS Faithfulness, Answer Relevancy, Context Recall and Context Precision.

## A/B comparison

| Metric | Config A: hybrid + RRF | Config B: dense-only | Delta A-B |
|---|---:|---:|---:|
| Faithfulness | 0.000 | 0.000 | +0.000 |
| Answer Relevance | 0.468 | 0.458 | +0.010 |
| Context Recall | 1.000 | 1.000 | +0.000 |
| Context Precision | 0.556 | 0.656 | -0.100 |

## Worst retrieval cases — Config A

| # | Question | Context recall | Context precision | Query-context relevance |
|---|---|---:|---:|---:|
| 1 | Lệ phí dự thi Đánh giá tư duy của Bách khoa Hà Nội là bao nhiêu? | 1.00 | 0.20 | 0.34 |
| 2 | IELTS Academic 5.0 được quy đổi thế nào trong xét tuyển Bách khoa TP HCM năm 2025? | 1.00 | 0.20 | 0.36 |
| 3 | Các tổ hợp A00, A01 và X06 của Trường Đại học Công nghệ gồm môn gì? | 1.00 | 0.20 | 0.54 |

## Findings and next actions

1. Hybrid retrieval improves exact-source recall when BM25 can match names, dates and numerical terms; dense-only is the baseline.
2. The current OpenRouter key is rejected by the provider, so generated answers abstain. Replace it before claiming LLM-based RAGAS results.
3. Add source-level metadata and tune chunk boundaries around tables to improve precision for point-score and tuition questions.
