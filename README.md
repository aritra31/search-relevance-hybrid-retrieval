# Search Relevance Optimization

I built this for a SurveyMonkey partnership project through Boston College (Nov 2025). Their help center search was basically text matching — users would type "how to create a form" and the article titled "Designing Your Survey" wouldn't show up. The relevant pages existed, they just weren't surfacing.

## What I found

I scraped 500+ help pages, wrote 78 labeled test queries by hand, and found the existing system failed on 45% of them. So I built a hybrid retrieval pipeline to fix it.

## Results

On the 78 golden queries (hand-labeled with expected URLs):

| Metric | Score |
|--------|-------|
| success@1 | 1.0 |
| success@3 | 1.0 |
| MRR | 1.0 |

On ~40 unseen queries (pages the model never trained against):

| Metric | Score |
|--------|-------|
| success@1 | 0.775 |
| success@3 | 1.0 |
| MRR | 0.88 |

Never misses the right article in the top 3, even on unseen queries.

## How it works

Each query runs through three retrievers in parallel — BM25 (lexical), TF-IDF with bigrams (cosine similarity), and fuzzy title matching via rapidfuzz. The three ranked lists get fused using Reciprocal Rank Fusion (k=60), then a semantic reranker rescores the top candidates using precomputed MiniLM-L6-v2 embeddings.

For ambiguous queries, there's a QPP (Query Performance Prediction) layer that checks how "clear" the query is before retrieval. If clarity is low, a local flan-t5-base model generates paraphrases and a feedback-refined variant, and all variants get fused together.

I originally tried a CrossEncoder (ms-marco-MiniLM) for reranking but it took 40-60 seconds per batch on my laptop. The precomputed embeddings approach is a dot product at query time - basically instant, no measurable accuracy loss on this corpus.

## Project structure
src/
config.py          — all tuning knobs (RRF k, reranker type, eval toggles)
crawler.py         — BFS crawler for help.surveymonkey.com
embedder.py        — precompute and cache MiniLM doc embeddings
pipeline.py        — retrieval, fusion, reranking, evaluation loop
reformulator.py    — QPP clarity scoring + flan-t5 query reformulation
tools/
split_golden.py    — train/test split of the golden dataset
make_unseen_golden.py — generates unseen eval set from unlabeled corpus pages
data/
corpus/articles.csv
SurveyMonkey_Golden_Dataset.xlsx
outputs/
docs/
SurveyMonkey_Technical_Report.pdf

## Running it

```bash
pip install -r requirements.txt

# crawl the help center (skips if articles.csv already exists)
python -m src.main --step crawl

# precompute document embeddings
python -m src.embedder

# run retrieval + evaluation
python -m src.main --step run --rewrite qppgen --split unseen
```

`--rewrite` controls variant generation: `rule` uses hand-written variants from the golden set, `local` always calls the LLM, `qppgen` only calls it when QPP says the query is ambiguous. `--split` picks which evaluation slice to run against.

## What I'd improve

The LLM reformulation doesn't move metrics much on these test sets because the queries are relatively clean. On real user logs with typos and mixed-intent queries ("ab test logic not working after randomization"), QPP would matter a lot more. I'd also want to test ColBERT as a middle ground between CrossEncoder accuracy and embedding speed.

## Stack

Python, rank-bm25, scikit-learn, sentence-transformers, rapidfuzz, transformers (flan-t5-base), BeautifulSoup, pandas
